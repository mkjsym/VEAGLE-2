# python eagle/ge_data/ge_data_all_llava_unified_topk.py --start 0 --end 67999 --outdir /data/dataset_cls_top100 --auto_distribute
import argparse
import os
import sys
import subprocess
import math
import time

# -----------------------------------------------------------------------------
# 1. Argument Parsing (Launcher & Worker 공통)
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='AirCache Data Generation')
parser.add_argument('--start', type=int, default=0, help='Total start index (or sub-start in worker mode)')
parser.add_argument('--end', type=int, default=100, help='Total end index (or sub-end in worker mode)')
parser.add_argument('--index', type=int, default=0, help='Sub-directory index for output')
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0], help='Specific GPU ID to use (worker mode)')
parser.add_argument('--outdir', type=str, default='outdir0', help='Output directory')
parser.add_argument('--auto_distribute', action='store_true', help='[Launcher Mode] Automatically split workload across all available GPUs')

args = parser.parse_args()

# -----------------------------------------------------------------------------
# 2. Launcher Logic (자동 분산 처리)
# -----------------------------------------------------------------------------
if args.auto_distribute:
    import torch
    
    # 가용 GPU 개수 확인
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("❌ No GPUs found. Exiting.")
        sys.exit(1)
        
    print(f"🚀 [Launcher] Found {num_gpus} GPUs. Distributing workload...")

    total_samples = args.end - args.start
    chunk_size = math.ceil(total_samples / num_gpus)
    
    processes = []
    
    for rank in range(num_gpus):
        # 각 GPU가 담당할 데이터 범위 계산
        sub_start = args.start + (rank * chunk_size)
        sub_end = min(args.start + ((rank + 1) * chunk_size), args.end)
        
        # 범위가 유효하지 않으면 스킵 (데이터가 GPU 수보다 적을 때)
        if sub_start >= sub_end:
            break

        # Worker 실행 명령어 구성
        # 자기 자신(__file__)을 호출하되, auto_distribute 옵션을 빼고 실행
        cmd = [
            sys.executable, __file__,
            '--start', str(sub_start),
            '--end', str(sub_end),
            '--index', str(rank),        # 각 GPU별로 다른 폴더(0, 1, 2...)에 저장하여 충돌 방지
            '--gpu_index', str(rank),    # 각 프로세스는 해당 rank의 GPU 1개만 할당받음
            '--outdir', args.outdir
        ]
        
        print(f"   [GPU {rank}] Processing indices {sub_start} ~ {sub_end} -> Saving to {args.outdir}/{rank}")
        
        # 비동기 실행 (subprocess)
        proc = subprocess.Popen(cmd)
        processes.append(proc)

    # 모든 작업이 끝날 때까지 대기
    exit_codes = [p.wait() for p in processes]
    
    if all(code == 0 for code in exit_codes):
        print(f"✅ [Launcher] All {len(processes)} jobs completed successfully.")
    else:
        print(f"⚠️ [Launcher] Some jobs failed. Exit codes: {exit_codes}")
        
    sys.exit(0)

# -----------------------------------------------------------------------------
# 3. Worker Logic (실제 데이터 처리)
# -----------------------------------------------------------------------------

# [중요] CUDA_VISIBLE_DEVICES 설정은 torch import 전에 해야 함
# Worker 모드에서는 gpu_index가 1개만 들어온다고 가정
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index[0])

import gc
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
from datasets import load_dataset, concatenate_datasets
import json
from fastchat.model.model_adapter import get_conversation_template
from PIL import Image

# 모델 경로 설정
bigname = "/data/youngmin/models/llava-1.5-7b-hf"

print(f"🔧 [Worker GPU {args.gpu_index[0]}] Initializing... Range: {args.start}-{args.end}")

def keep_topk_image_token(
    input_ids,
    loss_mask,
    hidden_states,
    image_features,
    attentions,
    img_tok_index=32000,
    topk=100,
):
    """
    input_ids: [1, seq_len]
    loss_mask: [1, seq_len]
    hidden_states: [1, seq_len, dim]
    attentions: list of [1, heads, seq_len, seq_len]
    image_features: [1, 576, dim] or [576, dim]
    """
    device = input_ids.device

    # 차원 축소
    input_ids = input_ids[0].to(device)        # [seq_len]
    loss_mask = loss_mask[0].to(device)        # [seq_len]
    hidden_states = hidden_states[0].to(device)  # [seq_len, dim]
    
    # CLS 토큰 인덱스 찾기
    cls_positions = (input_ids == img_tok_index).nonzero(as_tuple=True)[0]
    if cls_positions.numel() == 0:
        # CLS 토큰 없으면 원본 그대로
        return input_ids.unsqueeze(0), loss_mask.unsqueeze(0), hidden_states, image_features
    cls_index = cls_positions[0].item()

    # 마지막 레이어의 CLS어텐션 점수
    last_layer_attn = attentions[-1][0].to(device)  # [heads, seq_len, seq_len]
    # CLS 토큰이 attend한 각 토큰별 평균 점수
    attn_scores = last_layer_attn[:, cls_index, :].mean(dim=0)  # [seq_len]

    # 모든 이미지 토큰 위치
    image_token_indices = (input_ids == img_tok_index).nonzero(as_tuple=True)[0]

    # top-k 이미지 토큰 뽑기
    scores = attn_scores[image_token_indices].float()
    k = min(topk, scores.size(0))
    topk_local_idxs = torch.topk(scores, k).indices  # local indices
    topk_global_idxs = image_token_indices[topk_local_idxs]

    # CLS 인덱스도 추가
    topk_global = torch.cat([topk_global_idxs, torch.tensor([cls_index], device=device)])
    topk_global = torch.unique(topk_global)

    # 필터 마스크
    text_mask = input_ids != img_tok_index
    img_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    img_mask[topk_global] = True
    final_mask = text_mask | img_mask

    # 필터링
    filtered_input_ids = input_ids[final_mask].unsqueeze(0)
    filtered_loss_mask = loss_mask[final_mask].unsqueeze(0)
    filtered_hidden_states = hidden_states[final_mask]

    # 이미지 피처 필터링
    filtered_image_features = None
    if image_features is not None:
        feat = image_features[0] if image_features.dim() == 3 else image_features
        feat = feat.to(device)
        # topk_local_idxs는 CLS 제외한 순수 이미지 토큰이므로,
        # 실제 피처에서 뽑을 때는 local_idxs만 사용
        filtered_image_features = feat[topk_local_idxs]

    return filtered_input_ids, filtered_loss_mask, filtered_hidden_states, filtered_image_features

# -----------------------------------------------------------------------------
# AirCache Function Implementation
# -----------------------------------------------------------------------------
def keep_topk_image_token_aircache(
    input_ids,
    loss_mask,
    hidden_states,
    image_features,
    attentions,
    img_tok_index=32000,
    topk=100,
    alpha=0.9
):
    """
    AirCache: Activating Inter-modal Relevancy KV Cache Compression 구현
    [cite_start]Elite Observation Window 방식을 사용하여 이미지 토큰을 필터링합니다. [cite: 239]
    """
    device = input_ids.device

    input_ids = input_ids[0].to(device)
    loss_mask = loss_mask[0].to(device)
    hidden_states = hidden_states[0].to(device)
    
    # 마지막 레이어의 Attention Map 가져오기 (Heads 평균)
    last_layer_attn = attentions[-1][0].mean(dim=0).to(device) 

    # 1. 텍스트 토큰과 이미지 토큰 인덱스 구분
    is_img_token = (input_ids == img_tok_index)
    img_indices = is_img_token.nonzero(as_tuple=True)[0]
    text_indices = (~is_img_token).nonzero(as_tuple=True)[0]

    if img_indices.numel() == 0:
        return input_ids.unsqueeze(0), loss_mask.unsqueeze(0), hidden_states, image_features

    # [cite_start]2. Key Text Token Selection (Elite Observation Window 구성) [cite: 165]
    anchor_idx = input_ids.size(0) - 1 
    text_attn_scores = last_layer_attn[anchor_idx, text_indices] 

    # [cite_start]Eq (5): Relevance Threshold(alpha)를 이용한 필터링 [cite: 167]
    max_text_score = text_attn_scores.max()
    threshold_score = alpha * max_text_score
    
    key_text_mask = text_attn_scores >= threshold_score
    key_text_indices = text_indices[key_text_mask]

    # [cite_start]3. Visual Token Importance Assessment [cite: 178-179]
    attn_from_key_text = last_layer_attn[key_text_indices, :][:, img_indices]
    visual_scores = attn_from_key_text.mean(dim=0)

    # 4. Top-K 이미지 토큰 선정 및 필터링
    k = min(topk, visual_scores.size(0))
    topk_local_idxs = torch.topk(visual_scores, k).indices
    
    topk_global_idxs = img_indices[topk_local_idxs]
    topk_global_idxs, _ = torch.sort(topk_global_idxs)

    # 5. 마스크 생성 및 데이터 재구성
    keep_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    keep_mask[text_indices] = True
    keep_mask[topk_global_idxs] = True

    filtered_input_ids = input_ids[keep_mask].unsqueeze(0)
    filtered_loss_mask = loss_mask[keep_mask].unsqueeze(0)
    filtered_hidden_states = hidden_states[keep_mask]

    filtered_image_features = None
    if image_features is not None:
        feat = image_features[0] if image_features.dim() == 3 else image_features
        feat = feat.to(device)
        sorted_local_idxs, _ = torch.sort(topk_local_idxs)
        filtered_image_features = feat[sorted_local_idxs]

    return filtered_input_ids, filtered_loss_mask, filtered_hidden_states, filtered_image_features

# -----------------------------------------------------------------------------
# Dataset & Model Loading
# -----------------------------------------------------------------------------
def build_dataset_rank(tokenizer, split="train", select=None):
    # Processor 경로 수정 (bigname과 동일하게 설정)
    processor = AutoProcessor.from_pretrained(bigname)
    
    # 데이터셋 경로 (사용자 환경에 맞춤)
    ds1 = load_dataset('json', data_files="/data/llava_instruct_150k.json")[split]
    # ds1 이미지 경로: COCO
    ds1 = ds1.add_column('image_folder', ['/data/coco/train2017'] * len(ds1))
    
    ds2 = load_dataset('json', data_files="/data/sharegpt4v_instruct_gpt4-vision_cap100k.json")[split]
    # ds2 이미지 경로: COCO
    ds2 = ds2.add_column('image_folder', ['/data'] * len(ds2))
    
    # 병합 및 선택
    ds = concatenate_datasets([ds1, ds2]).shuffle(seed=41)
    
    # [Worker] 할당된 범위만 선택
    ds = ds.select(range(args.start, args.end))
        
    original_columns = ds.column_names
    num_proc = 4
    
    def contains_special_token(turn, tokenizer, special_token_id=32000):
        input_ids = tokenizer(turn).input_ids
        return special_token_id in input_ids

    def preprocess_function(examples):
        new_examples = {
            "conversation":[], "input_ids": [], "image": [], "pixel_values":[], "loss_mask": []
        }
        for i in range(len(examples['id'])):
            conv = get_conversation_template("vicuna")
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            sorce= examples['conversations'][i]
            
            if roles[sorce[0]["from"]] != conv.roles[0]:
                sorce = sorce[1:]
            conv.messages = []
            for j, sentence in enumerate(sorce):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversation=conv.get_prompt()
            
            image_file = examples['image'][i]
            folder = examples['image_folder'][i]
            try:
                image = Image.open(os.path.join(folder, image_file)).convert('RGB')
                inputs = processor(images=image, text=conversation, return_tensors="pt")
                input_ids=torch.as_tensor(inputs["input_ids"])[0]
                pixel_values=torch.as_tensor(inputs["pixel_values"])[0]
                loss_mask=torch.ones_like(input_ids)
                
                sep = conv.sep + conv.roles[1] + ": "
                turns = conversation.split(conv.sep2)
                
                cur_len = 1
                loss_mask[:cur_len] = 0
                for i, turn in enumerate(turns):
                    if turn == "": break
                    is_im_token = contains_special_token(turn,tokenizer)
                    turn_len = len(tokenizer(turn).input_ids)
                    if is_im_token : turn_len+=576

                    parts = turn.split(sep)
                    if len(parts) != 2: break
                    parts[0] += sep
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                    if is_im_token : instruction_len+=576
                    
                    if i==0: instruction_len -= 1
                    
                    loss_mask[cur_len: cur_len + instruction_len] = 0
                    cur_len += turn_len
                    if i==0: cur_len -= 1
                loss_mask[cur_len:] = 0
                
                new_examples["conversation"].append(conversation)
                new_examples["input_ids"].append(input_ids[None,:])
                new_examples["image"].append(image_file)
                new_examples["pixel_values"].append(pixel_values[None,:])
                new_examples["loss_mask"].append(loss_mask[None,:])
            except Exception as e:
                print(f"Skipping {image_file} due to error: {e}")
                continue

        return new_examples

    ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
        load_from_cache_file=False
    )

    ds.set_format(type="torch")
    return ds

# Tokenizer & Model Setup
# tokenizer 경로 수정 (bigname 사용)
bigtokenizer = AutoProcessor.from_pretrained(bigname).tokenizer
ds = build_dataset_rank(bigtokenizer)
print(f"[Worker GPU {args.gpu_index[0]}] Dataset loaded. Size: {len(ds)}")

# [수정 전]
bigmodel = LlavaForConditionalGeneration.from_pretrained(bigname, device_map="cuda", torch_dtype=torch.float16, attn_implementation="eager")

# [수정 후] 8-bit Quantization 적용
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_skip_modules=["mm_projector", "vision_tower"]  # 비전 관련 모듈은 정밀도 유지를 위해 fp16 유지 권장
# )

# bigmodel = LlavaForConditionalGeneration.from_pretrained(
#     bigname, 
#     device_map="cuda", 
#     quantization_config=bnb_config, # 8비트 설정 적용
#     # torch_dtype=torch.float16,    # 8bit 로드시에는 보통 자동 처리되므로 주석 처리하거나 놔둬도 무방
#     attn_implementation="eager"     # output_attentions=True를 위해 eager 유지
# )

bigmodel.eval()

# -----------------------------------------------------------------------------
# Main Generation Loop
# -----------------------------------------------------------------------------
@torch.no_grad()
def ge(data):
    input_ids = data["input_ids"]
    pixel_values = data["pixel_values"]
    loss_mask = data["loss_mask"]
    
    outs_big = bigmodel(input_ids.cuda(), pixel_values.cuda(), output_hidden_states=True, output_attentions=True)
    
    image_features = outs_big.image_hidden_states.cpu()
    hidden_state_big = outs_big.hidden_states[-1].cpu()
    
    # [AirCache Algorithm Applied]
    input_ids, loss_mask, hidden_state_big, image_features = keep_topk_image_token(
        input_ids, 
        loss_mask, 
        hidden_state_big, 
        image_features, 
        outs_big.attentions,
        img_tok_index=32000,
        topk=100
    )
    
    del outs_big
    gc.collect()
    torch.cuda.empty_cache()
    
    td = {
        "input_ids": input_ids.cpu()[0],
        "image": data["image"],
        "hidden_state": hidden_state_big.cpu(),
        "loss_mask": loss_mask.cpu()[0], 
        "image_features": image_features.cpu()
    }
    
    del hidden_state_big
    gc.collect()
    torch.cuda.empty_cache()
    
    return td

# Output directory setup
outdir_sub = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir_sub):
    try:
        os.makedirs(outdir_sub)
    except FileExistsError:
        pass

def writedata(name, data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length = len(os.listdir(name))
    idx = current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')

# Processing Loop 내부 수정
for data in tqdm(ds, desc=f"GPU {args.gpu_index[0]}"):
    # [추가] 시퀀스 길이 체크 (예: 3500 토큰 이상이면 스킵)
    # 24GB 메모리에서 output_attentions=True일 때 4096 풀 시퀀스는 터질 수 있음
    seq_len = data["input_ids"].shape[1]
    if seq_len > 3500:
        print(f"⚠️ Skipping data due to length ({seq_len} tokens) to avoid OOM.")
        continue

    try:
        with torch.no_grad():
            outdata = ge(data)
        
        writedata(outdir_sub, outdata)
        
        del outdata
    except torch.cuda.OutOfMemoryError:
        print(f"❌ OOM Error encountered at length {seq_len}. Skipping this sample.")
        torch.cuda.empty_cache() # 메모리 비우고 다음으로 진행
        continue
    except Exception as e:
        print(f"⚠️ Error processing data: {e}")
        continue
    
    gc.collect()
    torch.cuda.empty_cache()

print(f"✅ [Worker GPU {args.gpu_index[0]}] Finished processing.")
