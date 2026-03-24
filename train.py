"""
HSTU Training on MovieLens 20M -- config-matched to recsys-examples movielen_ranking.gin.

All hyperparameters match the original:
  batch_size=128, hidden_size=128, kv_channels=128, num_heads=4, num_layers=1,
  max_candidates=20, max_seq_len=200, prediction_head=[512,10], bf16,
  dropout=0.0, adam(lr=1e-3, beta1=0.9, beta2=0.98), seed=1234

Usage:
    python train.py                          # defaults match original gin
    python train.py --device cpu --nrows 500 # quick debug on CPU
"""

import argparse
import time

import torch
import torch_musa

from configs import HSTUConfig, RankingConfig, ShardedEmbeddingConfig, PositionEncodingConfig
from data.ml20m_dataset import ML20MDataset
from model.ranking_gr import RankingGR


def main():
    parser = argparse.ArgumentParser(description="HSTU ML-20M (gin-matched)")
    parser.add_argument("--dataset-path", type=str,
                        default="dataset/ml-20m/processed_seqs.csv")
    parser.add_argument("--device", type=str, default="musa")
    parser.add_argument("--backend", type=str, default="pytorch",
                        help="Attention: pytorch or triton")

    # --- Defaults match movielen_ranking.gin exactly ---
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--kv-channels", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--prediction-head", type=int, nargs="+", default=[512, 10])
    parser.add_argument("--hidden-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-train-iters", type=int, default=0,
                        help="0 = full epoch")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--no-bf16", action="store_true",
                        help="Disable bf16 (default: bf16 on)")
    parser.add_argument("--loss-log", type=str, default=None,
                        help="Save per-step loss to file")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_bf16 = not args.no_bf16 and device.type != "cpu"

    # --- Dataset (matches original: user_id contextual + movie_id + rating) ---
    print(f"Loading ML-20M from {args.dataset_path} ...")
    train_dataset = ML20MDataset(
        csv_path=args.dataset_path,
        batch_size=args.batch_size,
        max_history_seqlen=args.max_seq_len,
        max_num_candidates=args.max_candidates,
        num_tasks=args.num_tasks,
        is_train=True,
        shuffle=False,  # original gin: shuffle=False
        random_seed=args.seed,
        nrows=args.nrows,
    )
    print(f"Training: {train_dataset._num_samples} samples, "
          f"{len(train_dataset)} batches/epoch")

    eval_dataset = None
    if args.eval:
        eval_dataset = ML20MDataset(
            csv_path=args.dataset_path,
            batch_size=args.batch_size,
            max_history_seqlen=args.max_seq_len,
            max_num_candidates=args.max_candidates,
            num_tasks=args.num_tasks,
            is_train=False,
            shuffle=False,
            random_seed=args.seed,
            nrows=args.nrows,
        )
        print(f"Eval: {eval_dataset._num_samples} samples, "
              f"{len(eval_dataset)} batches")

    # --- Model (matches original gin + trainer/utils.py) ---
    HASH_SIZE = 200_000  # must match recsys-examples trainer/utils.py HASH_SIZE

    hstu_config = HSTUConfig(
        hidden_size=args.hidden_size,
        kv_channels=args.kv_channels,
        num_attention_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_dropout=args.hidden_dropout,
        is_causal=True,
        kernel_backend=args.backend,
        scaling_seqlen=args.max_seq_len,
        bf16=use_bf16,
        fp16=False,
        target_group_size=1,
        position_encoding_config=PositionEncodingConfig(
            num_position_buckets=8192,
            num_time_buckets=2048,
            use_time_encoding=False,
        ),
    )

    emb_configs = [
        ShardedEmbeddingConfig(
            feature_names=["user_id"],
            table_name="user_id",
            vocab_size=HASH_SIZE,
            dim=args.hidden_size,
        ),
        ShardedEmbeddingConfig(
            feature_names=["movie_id"],
            table_name="movie_id",
            vocab_size=HASH_SIZE,
            dim=args.hidden_size,
        ),
        ShardedEmbeddingConfig(
            feature_names=["rating"],
            table_name="action_weights",
            vocab_size=11,         # ratings 0-9 + padding
            dim=args.hidden_size,
        ),
    ]

    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=args.prediction_head,
        prediction_head_act_type="relu",
        prediction_head_bias=True,
        num_tasks=args.num_tasks,
    )

    model = RankingGR(hstu_config, task_config).to(device)
    if use_bf16:
        model._hstu_block.bfloat16()
        model._mlp.to(torch.bfloat16)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | BF16: {use_bf16}")
    print(f"Attention backend: {args.backend}")
    print(f"Parameters: {num_params:,}")
    print(f"Config: hidden={args.hidden_size} kv={args.kv_channels} "
          f"heads={args.num_heads} layers={args.num_layers} "
          f"head={args.prediction_head} dropout={args.hidden_dropout}")
    print(f"Optimizer: Adam(lr={args.lr}, beta=({args.adam_beta1},{args.adam_beta2}))")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    loss_log = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        step = 0
        t0 = time.time()

        for batch in train_dataset:
            if args.max_train_iters > 0 and step >= args.max_train_iters:
                break

            batch = batch.to(device)
            losses, (loss_det, logits, labels, _) = model(batch)
            loss = losses.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            total_loss += loss_val
            step += 1
            loss_log.append((epoch, step, loss_val))

            if step % args.log_interval == 0:
                avg = total_loss / step
                elapsed = time.time() - t0
                throughput = (step * args.batch_size) / elapsed
                print(f"Epoch {epoch+1} | Step {step}/{len(train_dataset)} | "
                      f"Loss: {loss_val:.6f} | Avg: {avg:.6f} | "
                      f"{throughput:.0f} samples/s")

        epoch_avg = total_loss / max(step, 1)
        elapsed = time.time() - t0
        print(f"--- Epoch {epoch+1} done | Steps: {step} | "
              f"Avg loss: {epoch_avg:.6f} | Time: {elapsed:.1f}s ---")

        if eval_dataset is not None:
            model.eval()
            eval_loss = 0.0
            eval_steps = 0
            with torch.no_grad():
                for batch in eval_dataset:
                    batch = batch.to(device)
                    losses, _ = model(batch)
                    eval_loss += losses.mean().item()
                    eval_steps += 1
            print(f"Eval loss: {eval_loss / max(eval_steps, 1):.6f}")

    if args.loss_log:
        with open(args.loss_log, "w") as f:
            f.write("epoch,step,loss\n")
            for e, s, l in loss_log:
                f.write(f"{e},{s},{l:.8f}\n")
        print(f"Loss log saved to {args.loss_log}")

    print("Training complete.")


if __name__ == "__main__":
    main()
