#!/bin/bash
python main.py \
    --train_dir data/new_rotate \
    --val_dir data/new_rotate \
    --output checkpoints \
    --batch_size 4 \
    --learning_rate 0.001 \
    --image_size 448 \
    --max_epochs 50 \
    --num_workers 4 \
    --checkpoints checkpoints/epoch=30-train_loss=0.0325-val_loss=0.0367.ckpt