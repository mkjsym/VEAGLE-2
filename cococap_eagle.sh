#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

CKPT_PATH=$1
TOK_PROCESS=$2
NUM_IMG_TOK=$3


echo "CKPT_PATH: ${CKPT_PATH}"
echo "TOK_PROCESS: ${TOK_PROCESS}"
echo "NUM_IMG_TOK: ${NUM_IMG_TOK}"

python -m model_vqa_eagle \
    --model-path llava-hf/llava-1.5-7b-hf \
    --ea-model-path ${CKPT_PATH} \
    --question-file /data2/coco_caption/coco_question.jsonl \
    --image-folder /data2/coco/val2014 \
    --answers-file ./coco_cap_t0_0.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --token-process ${TOK_PROCESS} \
    --num_img_tok ${NUM_IMG_TOK} \

#0:nothing
#1:remove
#2:pool
#3:remove_except_last
#4:remove_image_token_except_first
#5:keep_topk_image_token