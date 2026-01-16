import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["origin", "physical", "politics", "profanity", "age", "gender", "race", "religion"]

# 사용할 모델이 저장되어 있는 경로
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

while True:
    text = input("\n문장 입력 (종료하려면 q 입력): ").strip()
    if text.lower() == "q":
        break
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    print("\n[예측 결과]")
    for i, p in enumerate(probs):
        print(f"{LABELS[i]:<10} : {p:.3f}")
    preds = [LABELS[i] for i, p in enumerate(probs) if p >= 0.5]
    if preds:
        print("\n혐오로 판단된 유형:", ", ".join(preds))
    else:
        print("\n혐오 없음")
