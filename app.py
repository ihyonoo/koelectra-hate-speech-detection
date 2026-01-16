# app.py
# streamlit run app.py 명령어를 입력하면 실행됩니다.

import os
import html
import re
import uuid
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from streamlit.components.v1 import html as st_html


# 사용할 모델이 저장되어 있는 경로
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
THRESHOLD = 0.50    # 임계값(코드 고정)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"     # GPU or CPU
# 사용자가 이해하기 쉬운 라벨명 매핑
FRIENDLY_MAP = {
    "origin": "출신차별",
    "physical": "외모차별",
    "politics": "정치성향차별",
    "profanity": "혐오욕설",
    "age": "연령차별",
    "gender": "성차별",
    "race": "인종차별",
    "religion": "종교차별",
}
# id2label이 없거나 LABEL_0 형식일 때 사용할 기본 순서
FALLBACK_LIST = ["출신차별", "외모차별", "정치성향차별", "혐오욕설", "연령차별", "성차별", "인종차별", "종교차별"]
# 학습 라벨 순서가 위 FALLBACK_LIST와 다르면 여기서 직접 지정
OVERRIDE_BY_INDEX = None





# 모델 로드 (Streamlit 캐시) 
# 모델을 한 번만 Load하고 재사용하기 위해 캐시에 저장 -> 계속 모델을 새로 불러오면 비효율적이기 때문
# 앱이 다시 실행돼도 같은 세션에서는 모델을 다시 로드하지 않음
@st.cache_resource
def load_model():
    cfg = AutoConfig.from_pretrained(MODEL_DIR)         # Config Load
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)      # 토크나이저 Load
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, config=cfg).to(DEVICE).eval()   # 모델 Load

    # id2label 처리 (라벨 맵핑)
    id2label = None
    if hasattr(cfg, "id2label") and isinstance(cfg.id2label, dict) and len(cfg.id2label) > 0:
        try:
            id2label = {int(k): v for k, v in cfg.id2label.items()}
        except Exception:
            # 키가 이미 int일 수도 있음
            id2label = dict(cfg.id2label)

    
    num_labels = cfg.num_labels             # 라벨 개수

    return tok, mdl, id2label, num_labels   # 반환




# 라벨명 한국어 변환 함수
def friendly_label(i: int, id2label, num_labels: int) -> str:
    # 인덱스 강제 덮어쓰기
    if OVERRIDE_BY_INDEX and i < len(OVERRIDE_BY_INDEX):
        return OVERRIDE_BY_INDEX[i]

    # config.json의 id2label 기반
    if id2label is not None and i in id2label:
        raw = str(id2label[i]).strip()
        low = raw.lower()

        # "LABEL_0", "label-1", "Label 2" 같은 패턴은 인덱스 그대로 한국어 매핑
        m = re.fullmatch(r"label[\s_\-]?(\d+)", low)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < len(FALLBACK_LIST):
                return FALLBACK_LIST[idx]
            return f"라벨{idx}"

        # 영문 키워드 기반 맵핑
        if low in FRIENDLY_MAP:
            return FRIENDLY_MAP[low]
        for k, v in FRIENDLY_MAP.items():
            if k in low:  # politics-toxic 같은 변형 대비
                return v

        # 그래도 못 찾으면 원문 노출
        return raw

    # id2label이 없으면 인덱스 기반 Fallback
    if 0 <= i < len(FALLBACK_LIST):
        return FALLBACK_LIST[i]
    return f"라벨{i}"



# 예측 함수
@torch.no_grad()
def predict(text, tok, mdl):
    enc = tok(text, return_tensors="pt", truncation=True).to(DEVICE)
    prob = torch.sigmoid(mdl(**enc).logits)[0].cpu().tolist()
    return prob  # 0~1 사이 확률 리스트



# Streamlit UI
st.title("혐오 표현 자동 모자이크 시스템")
tok, mdl, id2label, num_labels = load_model()

txt = st.text_area(
    "댓글 입력",
    height=150,
    placeholder="댓글을 입력하세요. 혐오 표현으로 판단되면 모자이크 처리됩니다."
)


if st.button("작성"):
    if not txt.strip():
        st.warning("아무것도 입력되지 않았습니다.")
    else:
        # 1) 예측
        probs = predict(txt, tok, mdl)  # list[float], 길이=num_labels

        # 2) (라벨, 확률) 구성 + 한국어 변환
        labeled = [(friendly_label(i, id2label, num_labels), p) for i, p in enumerate(probs)]

        # 3) 임계값 이상 라벨만 확률 내림차순
        triggered = sorted([(n, p) for n, p in labeled if p >= THRESHOLD], key=lambda x: x[1], reverse=True)

        # 4) 출력
        if triggered:
            st.error(f"혐오 표현 감지됨")
            st.markdown("**모자이크 사유:**")
            st.markdown("\n".join([f"- {name} — **{round(p*100, 1)}%**" for name, p in triggered]))

            safe_text = html.escape(txt)
            uid = str(uuid.uuid4()).replace("-", "")
            box_id = f"mask_box_{uid}"
            text_id = f"mask_text_{uid}"
            hint_id = f"mask_hint_{uid}"

            st_html(f"""
                <style>
                  :root {{
                    --box-bg: #ffffff;   
                    --box-fg: #111111;  
                    --hint-bg: #00000066;
                    --hint-fg: #ffffff;
                    --border: #d0d0d0;  
                  }}
                  .blurred {{
                    filter: blur(8px);
                  }}
                </style>

                <div style="position:relative; margin-top:12px;">
                  <div id="{box_id}"
                       style="
                         transition: background-color 120ms ease-in-out, color 120ms ease-in-out;
                         padding: 14px 16px;
                         border: 1px solid var(--border);
                         border-radius: 10px;
                         cursor: pointer;
                         white-space: pre-wrap;
                         line-height: 1.6;
                         background: var(--box-bg);
                         color: var(--box-fg);
                         box-shadow: 0 1px 6px rgba(0,0,0,0.06);
                       "
                       title="클릭하여 보기/가리기">
                    <span id="{text_id}" class="blurred">{safe_text}</span>
                  </div>

                  <div id="{hint_id}"
                       style="
                         position:absolute; top:8px; right:12px;
                         font-size: 18px; opacity: 0.9;
                         background: var(--hint-bg);
                         color: var(--hint-fg);
                         padding: 3px 8px; border-radius: 6px;
                         user-select: none;
                         backdrop-filter: blur(2px);
                       ">
                       클릭해서 보기
                  </div>
                </div>

                <script>
                  (function() {{
                    const box  = document.getElementById("{box_id}");
                    const text = document.getElementById("{text_id}");
                    const hint = document.getElementById("{hint_id}");
                    let blurred = true;

                    if (box && text) {{
                      box.addEventListener("click", function() {{
                        blurred = !blurred;
                        if (blurred) {{
                          text.classList.add("blurred");
                          if (hint) hint.textContent = "클릭해서 보기";
                        }} else {{
                          text.classList.remove("blurred");
                          if (hint) hint.textContent = "다시 클릭해서 가리기";
                        }}
                      }});
                    }}
                  }})();
                </script>
            """, height=220)

        else:   # 혐오 발언 없을 경우
            st.success(f"혐오 표현 없음: 기준 {int(THRESHOLD*100)}% 미만으로 판단되어 모자이크 없이 표시합니다.")
            st.write(txt)
