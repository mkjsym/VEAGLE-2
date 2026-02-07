# python eagle/ge_data/ge_data_all_llava-mix-665k.py --start 0 --end 68000 --outdir /data/dataset_pretrain_llavamix_68000 --auto_distribute

import argparse
import os
import sys
import subprocess
import math
import json

# -----------------------------------------------------------------------------
# 1. Argument Parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='LLaVA Mix-665k AirCache Data Generation')
parser.add_argument('--start', type=int, default=0, help='Total start index')
parser.add_argument('--end', type=int, default=100, help='Total end index')
parser.add_argument('--index', type=int, default=0, help='Sub-directory index')
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0], help='Specific GPU ID (Worker mode)')
parser.add_argument('--outdir', type=str, default='/data/dataset_pretrain_llavamix_68000', help='Output directory')
parser.add_argument('--data_path', type=str, default='/data/llava-mix665k.jsonl')
parser.add_argument('--image_folder', type=str, default='/data/llava-mix665k-images') 
parser.add_argument('--auto_distribute', action='store_true', help='[Launcher] Auto split across all GPUs')

args = parser.parse_args()

# -----------------------------------------------------------------------------
# 2. Launcher Logic (자동 분산 처리)
# -----------------------------------------------------------------------------
if args.auto_distribute:
    import torch
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("❌ No GPUs found. Exiting.")
        sys.exit(1)
        
    print(f"🚀 [Launcher] Found {num_gpus} GPUs. Distributing workload ({args.start} ~ {args.end})...")

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
            '--data_path', args.data_path,
            '--image_folder', args.image_folder
        ]
        
        print(f"   [GPU {rank}] Processing {sub_start} ~ {sub_end}")
        proc = subprocess.Popen(cmd)
        processes.append(proc)

    exit_codes = [p.wait() for p in processes]
    
    if all(code == 0 for code in exit_codes):
        print(f"✅ [Launcher] All jobs completed successfully.")
    else:
        print(f"⚠️ [Launcher] Some jobs failed. Exit codes: {exit_codes}")
        
    sys.exit(0)

# -----------------------------------------------------------------------------
# 3. Worker Logic
# -----------------------------------------------------------------------------

# [핵심] GPU 격리 및 메모리 설정
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index[0])
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import Dataset
from fastchat.model.model_adapter import get_conversation_template
from PIL import Image

bigname = "llava-hf/llava-1.5-7b-hf"

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

    # 2. Key Text Token Selection
    anchor_idx = input_ids.size(0) - 1 
    text_attn_scores = last_layer_attn[anchor_idx, text_indices] 

    max_text_score = text_attn_scores.max()
    threshold_score = alpha * max_text_score
    
    key_text_mask = text_attn_scores >= threshold_score
    key_text_indices = text_indices[key_text_mask]

    # 3. Visual Token Importance Assessment
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

def contains_special_token(turn, tokenizer, special_token_id=32000):
    input_ids = tokenizer(turn).input_ids
    return special_token_id in input_ids

# -----------------------------------------------------------------------------
# Dataset & Model Loading
# -----------------------------------------------------------------------------
def build_dataset_rank(tokenizer):
    processor = AutoProcessor.from_pretrained(bigname)
    image_folder = args.image_folder
    
    print(f"[Worker GPU {args.gpu_index[0]}] Loading Mix-665k dataset from {args.data_path}...")
    
    try:
        with open(args.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        sys.exit(1)

    for item in raw_data:
        if 'id' in item:
            item['id'] = str(item['id'])

    ds = Dataset.from_list(raw_data)
    
    start_idx = args.start
    end_idx = min(args.end, len(ds))
    ds = ds.select(range(start_idx, end_idx))
    original_columns = ds.column_names
    
    num_proc = 1 
    
    def preprocess_function(examples):
        new_examples = {"conversation":[], "input_ids": [], "image": [], "pixel_values":[], "loss_mask": []}
        
        for i in range(len(examples['id'])):
            conv = get_conversation_template("vicuna")
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            source = examples['conversations'][i]
            
            if roles[source[0]["from"]] != conv.roles[0]:
                source = source[1:]
            
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                conv.append_message(role, sentence["value"])
            conversation = conv.get_prompt()
            
            image_file = examples['image'][i]
            full_image_path = os.path.join(image_folder, image_file)
            
            try:
                if not os.path.exists(full_image_path):
                    continue

                image = Image.open(full_image_path).convert('RGB')
                inputs = processor(images=image, text=conversation, return_tensors="pt")
                input_ids = torch.as_tensor(inputs["input_ids"])[0]
                pixel_values = torch.as_tensor(inputs["pixel_values"])[0]
                loss_mask = torch.ones_like(input_ids)
                
                sep = conv.sep + conv.roles[1] + ": "
                turns = conversation.split(conv.sep2)
                cur_len = 1
                loss_mask[:cur_len] = 0
                for k, turn in enumerate(turns):
                    if turn == "": break
                    is_im_token = contains_special_token(turn, tokenizer)
                    turn_len = len(tokenizer(turn).input_ids)
                    if is_im_token: turn_len += 576
                    
                    parts = turn.split(sep)
                    if len(parts) != 2: break
                    parts[0] += sep
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                    if is_im_token: instruction_len += 576
                    if k == 0: instruction_len -= 1
                    
                    loss_mask[cur_len: cur_len + instruction_len] = 0
                    cur_len += turn_len
                    if k == 0: cur_len -= 1
                loss_mask[cur_len:] = 0
                
                new_examples["conversation"].append(conversation)
                new_examples["input_ids"].append(input_ids[None,:])
                new_examples["image"].append(image_file)
                new_examples["pixel_values"].append(pixel_values[None,:])
                new_examples["loss_mask"].append(loss_mask[None,:])
                
            except Exception as e:
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

# --- Model & Generation Setup ---
bigtokenizer = AutoProcessor.from_pretrained(bigname).tokenizer
ds = build_dataset_rank(bigtokenizer)
print(f"[Worker GPU {args.gpu_index[0]}] Dataset prepared. Count: {len(ds)}")

gc.collect()
torch.cuda.empty_cache()

bigmodel = LlavaForConditionalGeneration.from_pretrained(
    bigname, 
    device_map="cuda", 
    torch_dtype=torch.float16,
    attn_implementation="eager"
)
bigmodel.eval()

# -----------------------------------------------------------------------------
# Main Generation Logic (Modified to match target format)
# -----------------------------------------------------------------------------
@torch.no_grad()
def ge(data):
    input_ids = data["input_ids"]
    pixel_values = data["pixel_values"]
    loss_mask = data["loss_mask"]
    
    # 1. Forward Pass with Attentions
    outs_big = bigmodel(input_ids.cuda(), pixel_values.cuda(), output_hidden_states=True, output_attentions=True)
    
    image_features = outs_big.image_hidden_states.cpu()
    hidden_state_big = outs_big.hidden_states[-1].cpu()
    
    # 2. AirCache Algorithm Applied (Uncommented as requested)
    # input_ids, loss_mask, hidden_state_big, image_features = keep_topk_image_token_aircache(
    #     input_ids, 
    #     loss_mask, 
    #     hidden_state_big, 
    #     image_features, 
    #     outs_big.attentions,
    #     img_tok_index=32000,
    #     topk=100,    
    #     alpha=0.9
    # )
    
    del outs_big
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. Construct Output Dictionary (Matched Format)
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

# -----------------------------------------------------------------------------
# Processing Loop (Matched Format with OOM Handling)
# -----------------------------------------------------------------------------
for data in tqdm(ds, desc=f"GPU {args.gpu_index[0]}"):
    # 시퀀스 길이 체크
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
        torch.cuda.empty_cache()
        continue
    except Exception as e:
        print(f"⚠️ Error processing data: {e}")
        continue
    
    gc.collect()
    torch.cuda.empty_cache()

print(f"✅ [Worker GPU {args.gpu_index[0]}] Finished processing.")
