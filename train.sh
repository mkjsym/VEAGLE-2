#!/bin/bash

accelerate launch -m --mixed_precision=bf16 eagle.train.main \
 --tmpdir /data/dataset_aircache_top100 \
 --cpdir /data/youngmin/checkpoints/aircache_100token_40epoch \
 --configpath /data/youngmin/legacy_sangjun/legacy_sangjun/EAGLE-LLAVA/eagle/train/llava-1.5_7B_config.json \
 --lr 1e-4 --bs 4 --epoch 40 --data_num 67999
