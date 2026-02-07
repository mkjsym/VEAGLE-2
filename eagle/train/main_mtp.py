import argparse
import warnings
import os
import json
from safetensors import safe_open
import torch
import torch.nn.functional as F  # [수정] Loss 계산을 위해 추가
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig
from accelerate import Accelerator
from accelerate.utils import set_seed
from typing import Any, Dict, List

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# [수정] list_files 함수 위치 유지
def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/data/youngmin/models/llava-1.5-7b-hf')
parser.add_argument('--configpath', type=str, default="config.json")
parser.add_argument('--pretrainedpath', type=str, default='/data/youngmin/legacy_sangjun/legacy_sangjun_data/ckpt/pretrain/state_50')
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--tmpdir', type=str, default='0')
parser.add_argument('--data_num', type=int, default=67999)
parser.add_argument('--cpdir', type=str, default='0')
# [수정] N-token prediction을 위한 인자 추가 (기본값 3)
parser.add_argument('--n_predict', type=int, default=3, help='Number of tokens to predict sequentially (MTP depth)')
args = parser.parse_args()

# [데이터 개수 자동 감지 로직 유지]
print(f"Scanning files in {args.tmpdir}...")
datapath = list_files(args.tmpdir)
real_data_num = len(datapath)

if real_data_num == 0:
    raise ValueError(f"No files found in {args.tmpdir}. Please check the --tmpdir path.")

print(f"Auto-detected data_num: {real_data_num}")
args.data_num = real_data_num
total_steps = int(args.data_num * 0.95 * (args.epoch + 1) / (args.bs * args.gradient_accumulation_steps))
warm_steps = total_steps // 100

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": args.epoch,
    "num_warmup_steps": warm_steps,
    "total_steps": total_steps,
    "p_w": 0.1,
    "v_w": 1.0,
    "head_w": 0.1,
    "num_workers": 8,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2048,
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5,
    "n_predict": args.n_predict # Config에 추가
}

# Accelerator 및 기본 설정
torch.backends.cuda.matmul.allow_tf32 = True
set_seed(0)
accelerator = Accelerator(mixed_precision='bf16',
                          gradient_accumulation_steps=train_config["gradient_accumulation_steps"])

# Model 관련 import (경로가 ..model 등 상대 경로이므로 환경에 맞게 동작한다고 가정)
from ..model.cnets import Model
from ..model.configs import EConfig

if accelerator.is_main_process:
    import wandb
    wandb.init(project="veagle-mtp", config=train_config) # project 이름 변경 예시

# [LM Head 로딩 로직 유지]
baseconfig = AutoConfig.from_pretrained(args.basepath).text_config
head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)
try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["language_model.lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path), framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice("language_model.lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].float()

head.weight.data = tensor
head.eval()
for param in head.parameters():
    param.requires_grad = False

# [Noise Class 및 Dataset Class 유지]
class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std
    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data

class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std
    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data

class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = torch.load(self.data[index], weights_only=True)
        new_data = {}
        # max_len 설정
        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]
        image_features = data["image_features"][:train_config["max_len"]][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        
        # 마지막 토큰 마스킹
        loss_mask[-1] = 0
        
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target
        new_data["image_features"] = image_features

        if self.transform:
            new_data = self.transform(new_data)
        return new_data

class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors
    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_image_features = torch.cat([item['image_features'] for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "image_features": batch_image_features,
        }
        return batch

def top_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

# [수정] 요청하신 MTP용 Loss 함수 (vloss 제거, ploss/rloss 사용)
def compute_loss(target_p, predict, loss_mask, topk=10):
    bsz, seq_len, vocab_size = target_p.shape
    
    # Head를 통과시켜 Logits 계산
    out_head = head(predict)
    
    # 마스크 적용
    masked_logits = out_head[loss_mask[..., 0] == 1]
    target_p_masked = target_p[loss_mask[..., 0] == 1]
    
    # 1. Probability Distance Loss (L1)
    predict_p = F.softmax(masked_logits, dim=-1)
    l1_distance = torch.abs(predict_p - target_p_masked)
    ploss = torch.mean(l1_distance.sum(dim=-1))

    # 2. Ranking Loss (Student가 Teacher의 Top-K 순서를 따르도록 유도)
    _, topk_indices = torch.topk(target_p_masked, k=topk, dim=-1)
    # Student Logits에서 Teacher의 Top-K 위치 값만 추출
    student_topk_logits = masked_logits.gather(-1, topk_indices)

    # Logcumsumexp를 이용한 Ranking Loss 계산
    reversed_logits = torch.flip(student_topk_logits, dims=[-1])
    log_cumsum_exp = torch.logcumsumexp(reversed_logits, dim=-1)
    log_denominator = torch.flip(log_cumsum_exp, dims=[-1])
    log_likelihood = student_topk_logits - log_denominator
    rloss = -torch.mean(log_likelihood.sum(-1))

    # 최종 Loss (가중치 적용)
    total_loss = 10 * ploss + 0.1 * rloss
    return total_loss, ploss, rloss, out_head

# [데이터셋 준비]
if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]

traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                          collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                          pin_memory=True)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                          collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

config = EConfig.from_pretrained(train_config["config_path"])
model = Model.from_pretrained(config, path=args.pretrainedpath)
# criterion = nn.SmoothL1Loss(reduction="none") # 새 Loss 함수에서는 사용하지 않음
optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

if train_config["is_warmup"]:
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)
    model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader, scheduler
    )
else:
    model, head, optimizer, train_loader, test_loader = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader
    )

# ==============================================================================
# Training Loop [수정] - Multi-token Prediction 적용
# ==============================================================================
for epoch in tqdm(range(num_epochs + 1)):
    # Metric 초기화
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    
    model.train()
    
    for batch_idx, data in enumerate(train_loader):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            
            # --- MTP 학습 루프 시작 ---
            curr_hidden = data["hidden_states"] # 초기 입력: t 시점의 hidden state
            curr_input_ids = data["input_ids"]  # 초기 입력: t 시점의 input_ids
            
            total_step_loss = 0
            n_predict = args.n_predict
            
            # Metric 기록용 (첫 번째 토큰 예측 결과만 로깅에 사용하거나 전체 평균 사용)
            first_step_out_head = None 
            first_step_target_head = None
            first_step_loss_mask = None

            for k in range(n_predict):
                # 1. Forward Pass
                # Model이 이전 step의 output(hidden state)을 받아 다음 hidden state 예측
                # image_features는 동일하게 사용 (또는 필요한 경우 처리)
                predict = model(
                    curr_hidden, 
                    input_ids=curr_input_ids, 
                    image_features=data["image_features"], 
                    attention_mask=data["attention_mask"]
                )
                
                # 2. Target 및 Mask 정렬 (Shift)
                # k=0일 때: predict는 t+1 예측. Target은 data["target"] (t+1부터 시작)
                # k=1일 때: predict는 t+2 예측. Target은 data["target"][:, 1:] (t+2부터 시작)
                # 길이를 맞추기 위해 slicing
                if k == 0:
                    step_target = data["target"]
                    step_loss_mask = data["loss_mask"][:, :, None] # [B, L, 1]
                else:
                    step_target = data["target"][:, k:, :]
                    step_loss_mask = data["loss_mask"][:, k:, None]
                    
                    # predict 길이도 target에 맞춰 잘라줌 (마지막 부분은 target이 없으므로)
                    predict = predict[:, :-k, :] 

                # 유효한 시퀀스 길이가 0이면 중단 (max_len 초과 시)
                if predict.size(1) == 0:
                    break

                # 3. Teacher Distribution 계산 (On-the-fly)
                with torch.no_grad():
                    target_head = head(step_target)
                    target_p = nn.Softmax(dim=2)(target_head)
                    target_p = target_p.detach()

                # 4. Loss 계산
                step_loss, step_ploss, step_rloss, out_head = compute_loss(target_p, predict, step_loss_mask)
                
                # Loss 누적 (모든 step의 loss 합산)
                total_step_loss += step_loss

                # 첫 번째 스텝의 결과만 Metric 기록용으로 저장 (가장 중요하므로)
                if k == 0:
                    first_step_out_head = out_head
                    first_step_target_head = target_head
                    first_step_loss_mask = step_loss_mask
                    # 다음 스텝을 위해 input 업데이트
                    # Student의 예측(predict)을 다음 스텝의 입력으로 사용 (Recurrent)
                    curr_hidden = predict
                    # Input IDs도 시프트 (Positional Embedding 등을 위해)
                    curr_input_ids = curr_input_ids[:, 1:]
                else:
                    # k > 0 인 경우, 다음 루프를 위해 hidden 업데이트
                    # predict는 이미 슬라이싱 되었으므로, 다음 iter에서 target은 더 많이 슬라이싱됨
                    curr_hidden = predict
                    curr_input_ids = curr_input_ids[:, 1:]

            # --- MTP 루프 종료 ---

            accelerator.backward(total_step_loss)
            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            if is_warmup:
                scheduler.step()

            # --- Logging (첫 번째 Step 기준) ---
            with torch.no_grad():
                # 정확도 계산은 k=0 (t+1 예측) 기준으로 수행
                _, predicted = torch.max(first_step_out_head, 2)
                _, target_idx = torch.max(first_step_target_head, 2)
                
                ct = first_step_loss_mask.sum().item()
                cc = ((predicted == target_idx) * first_step_loss_mask.squeeze()).sum().item()
                
                masked_out = first_step_out_head.view(-1, vocab_size)[first_step_loss_mask.view(-1) == 1]
                masked_target = target_idx.view(-1)[first_step_loss_mask.view(-1) == 1]
                
                topkacc = top_accuracy(masked_out, masked_target, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                
                total += ct
                correct += cc

            if accelerator.is_main_process and ct != 0:
                # Loss는 전체 Step 합산 Loss 로깅
                wandb.log({
                    "train/lr": optimizer.optimizer.param_groups[0]["lr"],
                    "train/loss": total_step_loss.item(),
                    "train/acc": cc / ct, # Step 1 Accuracy
                })
            
            epoch_loss += total_step_loss.item()
            num_batches += 1

    # Epoch 단위 Logging 및 Sync
    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
        wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})

    # ==========================================================================
    # Evaluation Loop [수정] - Test에서도 MTP 로직 적용 가능하나, 보통 Acc 측정은 1-step 기준
    # 여기서는 Loss 계산 방식을 Train과 동일하게 맞춰줌
    # ==========================================================================
    if (epoch + 1) % train_config["save_freq"] == 0:
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.eval()

        # k_acc = [[] for i in range(5)] # getkacc는 기존 함수 사용 (inference simulation)

        for batch_idx, data in enumerate(test_loader):
            with torch.no_grad():
                # (Optional) Generate Accuracy 측정은 기존 getkacc 함수 사용 (단일 step forward 반복)
                # if batch_idx < 10: ... (기존 코드 유지 가능)

                # MTP Loss Evaluation
                curr_hidden = data["hidden_states"]
                curr_input_ids = data["input_ids"]
                total_step_loss = 0
                n_predict = args.n_predict
                
                # Logging용 변수
                first_step_out_head = None
                first_step_target_head = None
                first_step_loss_mask = None

                for k in range(n_predict):
                    predict = model(curr_hidden, input_ids=curr_input_ids, image_features=data["image_features"], attention_mask=data["attention_mask"])
                    
                    if k == 0:
                        step_target = data["target"]
                        step_loss_mask = data["loss_mask"][:, :, None]
                    else:
                        step_target = data["target"][:, k:, :]
                        step_loss_mask = data["loss_mask"][:, k:, None]
                        predict = predict[:, :-k, :]

                    if predict.size(1) == 0: break

                    target_head = head(step_target)
                    target_p = nn.Softmax(dim=2)(target_head)
                    
                    step_loss, _, _, out_head = compute_loss(target_p, predict, step_loss_mask)
                    total_step_loss += step_loss

                    if k == 0:
                        first_step_out_head = out_head
                        first_step_target_head = target_head
                        first_step_loss_mask = step_loss_mask
                        curr_hidden = predict
                        curr_input_ids = curr_input_ids[:, 1:]
                    else:
                        curr_hidden = predict
                        curr_input_ids = curr_input_ids[:, 1:]

                # Accuracy (Step 1 기준)
                _, predicted = torch.max(first_step_out_head, 2)
                _, target_idx = torch.max(first_step_target_head, 2)
                ct = first_step_loss_mask.sum().item()
                cc = ((predicted == target_idx) * first_step_loss_mask.squeeze()).sum().item()
                
                masked_out = first_step_out_head.view(-1, vocab_size)[first_step_loss_mask.view(-1) == 1]
                masked_target = target_idx.view(-1)[first_step_loss_mask.view(-1) == 1]
                
                topkacc = top_accuracy(masked_out, masked_target, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                
                total += ct
                correct += cc
                epoch_loss += total_step_loss.item()
                num_batches += 1

        # Test Metric Logging (기존 코드와 동일 구조)
        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        
        epoch_loss /= num_batches
        if accelerator.is_local_main_process:
            print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
            wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
            accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
