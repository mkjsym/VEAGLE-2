# python eagle/ge_data/ge_data_all_ShareGPT.py --start 0 --end 68000 --outdir /data/dataset_ShareGPT_68000 --auto_distribute

import argparse
import os
import sys
import subprocess
import math
import time
import json
import gc
import copy
import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
from fastchat.model.model_adapter import get_conversation_template

# -----------------------------------------------------------------------------
# 1. Argument Parsing (Launcher & Worker 공통)
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='ShareGPT Data Generation (Text-Only)')
parser.add_argument('--start', type=int, default=0, help='Total start index')
parser.add_argument('--end', type=int, default=100, help='Total end index')
parser.add_argument('--index', type=int, default=0, help='Sub-directory index')
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0], help='Specific GPU ID')
parser.add_argument('--outdir', type=str, default='outdir_sharegpt', help='Output directory')
parser.add_argument('--data_path', type=str, default="/data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json", help="ShareGPT JSON path")
parser.add_argument('--auto_distribute', action='store_true', help='Automatically split workload')

args = parser.parse_args()

# -----------------------------------------------------------------------------
# 2. Launcher Logic (자동 분산 처리 - 변경 없음)
# -----------------------------------------------------------------------------
if args.auto_distribute:
    import torch
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("❌ No GPUs found. Exiting.")
        sys.exit(1)
        
    print(f"🚀 [Launcher] Found {num_gpus} GPUs. Distributing workload...")

    total_samples = args.end - args.start
    chunk_size = math.ceil(total_samples / num_gpus)
    
    processes = []
    
    for rank in range(num_gpus):
        sub_start = args.start + (rank * chunk_size)
        sub_end = min(args.start + ((rank + 1) * chunk_size), args.end)
        
        if sub_start >= sub_end:
            break

        cmd = [
            sys.executable, __file__,
            '--start', str(sub_start),
            '--end', str(sub_end),
            '--index', str(rank),
            '--gpu_index', str(rank),
            '--outdir', args.outdir,
            '--data_path', args.data_path
        ]
        
        print(f"   [GPU {rank}] Processing {sub_start} ~ {sub_end} -> {args.outdir}/{rank}")
        proc = subprocess.Popen(cmd)
        processes.append(proc)

    exit_codes = [p.wait() for p in processes]
    if all(code == 0 for code in exit_codes):
        print(f"✅ [Launcher] All jobs completed.")
    else:
        print(f"⚠️ [Launcher] Some jobs failed: {exit_codes}")
    sys.exit(0)

# -----------------------------------------------------------------------------
# 3. Worker Logic (ShareGPT Text Processing)
# -----------------------------------------------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index[0])

# 모델 경로 설정
bigname = "/data/youngmin/models/llava-1.5-7b-hf" 

print(f"🔧 [Worker GPU {args.gpu_index[0]}] Initializing ShareGPT Processing...")

def build_dataset_rank(tokenizer, split="train"):
    # ShareGPT 데이터셋 로드
    print(f"Loading dataset from: {args.data_path}")
    ds = load_dataset('json', data_files=args.data_path)
    ds = ds['train']
    
    # 범위 선택
    start_idx = max(0, args.start)
    end_idx = min(len(ds), args.end)
    ds = ds.select(range(start_idx, end_idx))
    
    original_columns = ds.column_names
    num_proc = 4

    def preprocess_function(examples):
        new_examples = {
            "conversation": [],
            "input_ids": [],
            "loss_mask": []
        }
        
        for i in range(len(examples['conversations'])):
            try:
                # 1. Conversation Template (Vicuna)
                conv = get_conversation_template("vicuna")
                roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
                source = examples['conversations'][i]

                if roles.get(source[0]["from"]) != conv.roles[0]:
                    source = source[1:]

                conv.messages = []
                for j, sentence in enumerate(source):
                    role = roles.get(sentence["from"])
                    if role:
                        conv.append_message(role, sentence["value"])
                
                conversation = conv.get_prompt()

                # 2. Tokenize (Text Only)
                inputs = tokenizer(conversation, return_tensors="pt")
                input_ids = inputs.input_ids[0]
                
                # 3. Loss Mask Calculation
                loss_mask = torch.ones_like(input_ids)
                sep = conv.sep + conv.roles[1] + ": "
                turns = conversation.split(conv.sep2)
                
                # BOS 처리 (Llama tokenizer dependent)
                cur_len = 1
                if input_ids[0] != tokenizer.bos_token_id:
                     cur_len = 0
                loss_mask[:cur_len] = 0 

                for k, turn in enumerate(turns):
                    if turn == "": break
                    
                    turn_len = len(tokenizer(turn).input_ids)
                    if tokenizer.bos_token_id and tokenizer(turn).input_ids[0] == tokenizer.bos_token_id:
                        turn_len -= 1

                    parts = turn.split(sep)
                    if len(parts) != 2: break
                    
                    # User Instruction Masking
                    parts[0] += sep
                    instruction_len = len(tokenizer(parts[0]).input_ids)
                    if tokenizer.bos_token_id and tokenizer(parts[0]).input_ids[0] == tokenizer.bos_token_id:
                        instruction_len -= 1
                    
                    # k==0일 때의 미세 조정 (상황에 따라 다를 수 있음, 일반적 Vicuna 포맷)
                    if k == 0: 
                        instruction_len -= 1

                    mask_end = min(cur_len + instruction_len, len(loss_mask))
                    loss_mask[cur_len : mask_end] = 0
                    
                    cur_len += turn_len
                    if k == 0: cur_len -= 1 # 첫 턴 보정
                    
                if cur_len < len(loss_mask):
                    loss_mask[cur_len:] = 0

                new_examples["conversation"].append(conversation)
                new_examples["input_ids"].append(input_ids[None, :])
                new_examples["loss_mask"].append(loss_mask[None, :])
                
            except Exception as e:
                print(f"Error in preprocessing: {e}")
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

# --- Model & Dataset Load ---
processor = AutoProcessor.from_pretrained(bigname)
bigtokenizer = processor.tokenizer

ds = build_dataset_rank(bigtokenizer)
print(f"[Worker GPU {args.gpu_index[0]}] Dataset loaded. Size: {len(ds)}")

# LLaVA 모델 로드 (Text-only 처리용)
bigmodel = LlavaForConditionalGeneration.from_pretrained(
    bigname, 
    device_map="cuda", 
    torch_dtype=torch.float16,
    attn_implementation="eager"
)
bigmodel.eval()

# -----------------------------------------------------------------------------
# Main Generation Loop
# -----------------------------------------------------------------------------
@torch.no_grad()
def ge(data):
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    
    # Text-only: pixel_values는 None 처리
    # output_attentions=True는 유지하지만 AirCache는 사용하지 않으므로 사실상 불필요할 수 있음
    # 하지만 데이터 일관성을 위해 hidden_state 추출 방식은 유지
    outs_big = bigmodel(
        input_ids.cuda(), 
        pixel_values=None, 
        output_hidden_states=True
    )
    
    hidden_state_big = outs_big.hidden_states[-1].cpu()
    
    # AirCache 제거됨: 이미지 토큰이 없으므로 원본 그대로 저장
    # image_features는 None
    
    del outs_big
    gc.collect()
    torch.cuda.empty_cache()
    
    td = {
        "input_ids": input_ids.cpu()[0],
        "image": None,  # 이미지가 없음
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": loss_mask.cpu()[0],
        "image_features": None # 이미지가 없음
    }
    
    return td

# Output Dir
outdir_sub = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir_sub):
    os.makedirs(outdir_sub, exist_ok=True)

def writedata(name, data_point):
    current_length = len(os.listdir(name))
    idx = current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')

# Loop
for data in tqdm(ds, desc=f"GPU {args.gpu_index[0]}"):
    seq_len = data["input_ids"].shape[1]
    
    # 텍스트 전용이므로 4096 꽉 채우면 OOM 가능성 있음 (안전하게 3500 컷 유지)
    if seq_len > 3500:
        print(f"⚠️ Skipping long sequence ({seq_len})")
        continue

    try:
        with torch.no_grad():
            outdata = ge(data)
        
        writedata(outdir_sub, outdata)
        del outdata

    except torch.cuda.OutOfMemoryError:
        print(f"❌ OOM Error. Skipping.")
        torch.cuda.empty_cache()
        continue
    except Exception as e:
        print(f"⚠️ Error: {e}")
        continue
    
    gc.collect()
    torch.cuda.empty_cache()

print(f"✅ [Worker GPU {args.gpu_index[0]}] Finished.")
