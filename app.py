import os
import gdown
import pandas as pd
import numpy as np
import faiss
import streamlit as st
from openai import OpenAI

# ==========================================
# 1. 환경 변수에서 OpenAI API 키 불러오기
# ==========================================
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    st.error("OpenAI API Key가 설정되지 않았습니다. Streamlit Secrets 또는 환경 변수에서 OPENAI_API_KEY를 설정하세요.")
    st.stop()

client = OpenAI(api_key=API_KEY)

# ==========================================
# 2. 구글 드라이브에서 데이터 파일 다운로드
# ==========================================
INDEX_FILE_ID = "10O9D9kIHHbRPMJN_52mPCSjYklgBZpMJ"  # 드라이브에서 추출한 파일 ID
CSV_FILE_ID = "1HpNAK0vO11XiifJexX7t3Ly902WSGKEJ"    # 드라이브에서 추출한 파일 ID

index_file = "slowletter_entities.index"
csv_file = "slowletter_full_with_entities.csv"

if not os.path.exists(index_file):
    st.write("Index 파일을 다운로드 중입니다...")
    gdown.download(f"https://drive.google.com/uc?id={INDEX_FILE_ID}", index_file, quiet=False)

if not os.path.exists(csv_file):
    st.write("CSV 파일을 다운로드 중입니다...")
    gdown.download(f"https://drive.google.com/uc?id={CSV_FILE_ID}", csv_file, quiet=False)

# ==========================================
# 3. 데이터와 벡터 인덱스 로드 (캐싱)
# ==========================================
@st.cache_resource
def load_resources():
    index = faiss.read_index(index_file)
    df = pd.read_csv(csv_file)
    return index, df

index, df = load_resources()

# ==========================================
# 4. 검색 함수
# ==========================================
def search_answer(query, top_k=5):
    # 질문 임베딩
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    emb = np.array(emb).astype("float32").reshape(1, -1)

    # 벡터 검색
    D, I = index.search(emb, top_k)
    rows = df.iloc[I[0]]

    # 검색 컨텍스트 생성
    context = "\n\n".join(
        f"{r.get('date','')}\n제목: {r.get('h3_title','')}\n본문: {r.get('h3_content_html','')}\n엔티티:{r.get('entities','')}\n이벤트:{r.get('events','')}"
        for _, r in rows.iterrows()
    )

    # GPT 답변 생성
    prompt = f"""
당신은 '슬로우레터(Slowletter)' 스타일의 뉴스 분석 어시스턴트입니다.

역할:
- 주어진 검색 문서(맥락)와 질문을 기반으로, 한국어로 간결하고 깊이 있게 설명합니다.
- 답변은 슬로우레터 특유의 **짧은 문장, 맥락 강조, 불필요한 군더더기 없는 문체**로 작성합니다.
- 팩트와 맥락 중심. 과장이나 추측은 피합니다.

작성 규칙:
1. **출력 형식**:
   - 첫 줄에 핵심 요약 (짧고 단정적인 한 문장)
   - 이후 불릿 포인트(`•`)로 맥락/세부 설명을 나눠 주세요.
   - 가능한 경우 문서 출처(제목 또는 날짜)를 괄호로 표기해 주세요.

2. **참고한 문서**
   - 마지막에 "참고: ○○(날짜)" 형태로 관련 문서를 표기해 주세요.
   - 출처가 여러 개면 쉼표로 구분.

3. 스타일:
   - 건조하고 간결한 어조
   - 인용은 원문 표현을 그대로 사용

다음은 질문과 검색된 문서들입니다.

질문:
{query}

검색된 문서:
{context}
"""

    ans = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

    return ans

# ==========================================
# 5. Streamlit UI
# ==========================================
st.title("Slow News Insight Bot.")

query = st.text_area("슬로우뉴스에 물어보세요.")

if st.button("실행."):
    if query.strip():
        with st.spinner("분석을 시작합니다..."):
            answer = search_answer(query)
            st.write("### 답변")
            st.write(answer)
