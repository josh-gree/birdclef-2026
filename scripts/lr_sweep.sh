#!/usr/bin/env bash
set -e

FIXED="--total-val-rounds 10 --val-every-epochs 0.5 --soundscape-repeat 10 --dropout 0.5 --hidden 1028 --weight-decay 0.0001 --unfreeze-blocks 2 --use-augmentation --detach --batch-size 32"

# EfficientNet: head lr x backbone lr
for lr in 1e-4 3e-4 1e-3 3e-3; do
  for blr in 1e-5 1e-4; do
    uv run birdclef run birdclef_baseline $FIXED --backbone efficientnet_b3 --lr $lr --backbone-lr $blr
  done
done

# ViT: head lr x backbone lr
for lr in 1e-5 1e-4 3e-4 1e-3; do
  for blr in 1e-6 1e-5; do
    uv run birdclef run birdclef_baseline $FIXED --backbone vit_base --lr $lr --backbone-lr $blr
  done
done
