#train.py

import os, csv, argparse, math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score
from tqdm import tqdm


# Labels
LABEL_NAMES = [
    "origin",    # 0: 출신
    "physical",  # 1: 외모
    "politics",  # 2: 정치
    "profanity", # 3: 욕설
    "age",       # 4: 연령
    "gender",    # 5: 성별
    "race",      # 6: 인종
    "religion"   # 7: 종교
]
NOT_HATE_LABEL = 8  # 비혐오는 8, 학습에는 반영하지 않고 라벨이 없는 것으로 간주


# 하이퍼파라미터 컨테이너
@dataclass
class Args:
    train: str        # 학습 데이터 경로
    valid: str        # 검증 데이터 경로
    out: str          # 출력(모델 저장) 경로
    model: str        # 사용할 사전학습 모델 이름
    max_len: int      # 입력 문장 최대 길이
    batch: int        # 배치 크기
    epochs: int       # 학습 반복 횟수
    lr: float         # 학습률
    grad_accum: int   # 그래디언트 누적 스텝 수
    fp16: bool        # FP16 혼합정밀 학습 여부


# Read data
# 파일 포맷: document<TAB>label
# 모델이 학습할 수 있도록 [(문장, [라벨인덱스들]), ...] 형태로 반환하는 함수
def read_kmhas_txt(path):
    rows = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:    # File open
        
        reader = csv.reader(f, delimiter="\t", quotechar='"')   # Tap으로 구분된 TSV 파일을 읽음, 따옴표는 자동으로 제거
        _ = next(reader, None)  # 헤더 스킵

        for line in reader:     # 본문데이터 한 줄씩 반복
            # 빈줄 스킵
            if not line:
                continue
            
            # 문장,라벨 분리
            if len(line) == 1:
                text, lab = line[0], ""
            else:
                text, lab = line[0], line[1]

            # 문자열 형태의 라벨을 숫자열로 변환
            lab = lab.strip()   # 공백 제거
            labels = []
            if lab:
                for tok in lab.split(","):
                    tok = tok.strip()
                    if tok.isdigit():
                        labels.append(int(tok))

            rows.append((text, labels))     # 문장과 라벨을 튜플로 저장

    return rows



# 데이터셋 정의
class KMHasDataset(Dataset):
    # 초기화
    def __init__(self, rows, tokenizer, max_len: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    # 데이터셋의 개수 반환
    def __len__(self):
        return len(self.rows)

    # 인덱스로 하나의 {문장과 라벨 리스트} 가져오기
    def __getitem__(self, idx):
        text, labels = self.rows[idx]

        enc = self.tok(         # 문장 토크나이징
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        # 라벨 멀티핫 벡터 생성
        y = np.zeros(len(LABEL_NAMES), dtype=np.float32)

        for lid in labels:
            if 0 <= lid < len(LABEL_NAMES):  # 0~7만 멀티핫
                y[lid] = 1.0
       
        item = {k: v.squeeze(0) for k, v in enc.items()}        # 토크나이즈 결과 정리
        item["labels"] = torch.tensor(y, dtype=torch.float32)   # 라벨 추가

        return item




# 모델 평가 함수
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch in loader:
            # 라벨은 나중에 따로 꺼내서 CPU로 모음
            labels = batch["labels"].cpu().numpy()

            batch = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            ).logits

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            y_true.append(labels)
            y_pred.append(preds)
            y_prob.append(probs)

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    y_prob = np.vstack(y_prob)

    # 1) F1 macro / micro / weighted
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # 2) Exact Match Ratio (E.M.)
    exact_match = np.mean(np.all(y_true == y_pred, axis=1))

    # 3) Hamming Loss (H.L.)
    hl = hamming_loss(y_true, y_pred)

    # 4) AUC (macro)
    try:
        auc = roc_auc_score(y_true, y_prob, average="macro")
    except ValueError:
        # 어떤 라벨이 전부 0/1만 나오는 경우 발생 가능
        auc = 0.0

    return f1_macro, f1_micro, f1_weighted, exact_match, auc, hl




# Train
def train_loop(args: Args):

    # CPU, GPU 선택
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] device = {device}")

    # Read data
    train_rows = read_kmhas_txt(args.train)
    valid_rows = read_kmhas_txt(args.valid)
    print(f"[Info] train samples = {len(train_rows)} | valid samples = {len(valid_rows)}")

    # Dataset, DataLoader 준비
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds = KMHasDataset(train_rows, tokenizer, args.max_len)
    valid_ds = KMHasDataset(valid_rows, tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=max(1, args.batch*2), shuffle=False, drop_last=False)

    # 모델 생성
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABEL_NAMES),
        problem_type="multi_label_classification"
    ).to(device)

    # Optimizer / Scheduler / Loss 정의
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = (len(train_loader) // max(1, args.grad_accum)) * max(1, args.epochs)
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=0, num_training_steps=max(1, total_steps)
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    # 혼합정밀(AMP) 스케일러
    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device == "cuda"))

    # 학습 준비
    best_macro = -1.0
    os.makedirs(args.out, exist_ok=True)

    # 학습 루프 (에포크 반복)
    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}")
        optim.zero_grad(set_to_none=True)

        # 미니배치 반복
        for it, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(args.fp16 and device == "cuda")):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                ).logits
                loss = criterion(logits, batch["labels"])

            # 역전파 & 옵티마이저 스텝
            scaler.scale(loss).backward()
            if it % args.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()

            # 진행률 표시, 50 스텝마다 현재 손실값을 진행률바에 표시
            if it % 50 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 검증 단계, 6개 지표 계산 
        macro, micro, weighted, em, auc, hl = evaluate(model, valid_loader, device, threshold=0.5)
        print(
            f"[VAL] F1-macro={macro:.4f} | F1-micro={micro:.4f} | "
            f"F1-weighted={weighted:.4f}"
        )
        print(
            f"[VAL] EM={em:.4f} | AUC={auc:.4f} | H.L.={hl:.4f}"
        )

        # Save best (macro F1)
        if macro > best_macro:
            best_macro = macro
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
            print(f"[Save] Best model saved to: {args.out} (macro={best_macro:.4f})")


    # 종료
    print("[Done] Training complete.")




# Main
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)

    # train 데이터셋
    train_path = os.path.join(BASE_DIR, "dataset", "train", "kmhas_train.txt")
    # val 데이터셋
    valid_path = os.path.join(BASE_DIR, "dataset", "val", "kmhas_valid.txt")
    # best 모델 저장할 경로
    out_path   = os.path.join(BASE_DIR, "models")

    # 하이퍼파라미터 설정
    args = Args(
        train=train_path,
        valid=valid_path,
        out=out_path,
        model="monologg/koelectra-base-v3-discriminator",
        max_len=128,
        batch=64,
        epochs=5,
        lr=3e-5,
        grad_accum=1,
        fp16=True
    )

    # 학습 시작
    train_loop(args)