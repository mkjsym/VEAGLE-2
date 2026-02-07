#!/bin/bash

accelerate launch -m --mixed_precision=bf16 eagle.train.main \
 --tmpdir /data/dataset_aircache_top20_259736 \
 --cpdir /data/youngmin/checkpoints/aircache_20token_259736_40epoch \
 --configpath /data/llava-1.5_7B_config.json \
 --lr 1e-4 --bs 4 --epoch 40 --data_num 259736
