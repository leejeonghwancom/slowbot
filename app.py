import os
import gdown
import pandas as pd
import numpy as np
import faiss
import streamlit as st
from openai import OpenAI

# ==========================================
# 1. 환경 변수 (API KEY)
# ==========================================
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    st.error("OpenAI API Key가 없습니다. secrets 또는 환경 변수로 OPENAI_API_KEY를 설정하세요.")
    st.stop()

client = OpenAI(api_key=API_KEY)

# ==========================================
# 2. Google Drive에서 파일 다운로드
# ==========================================
INDEX_FILE_ID = "10O9D9kIHHbRPMJN_52mPCSjYklgBZpMJ"
CSV_FILE_ID = "1HpNAK0vO11XiifJexX7t3Ly902WSGKEJ"

index_file = "slowletter_entities.index"
csv_file = "slowletter_full_with_entities.csv"

if not os.path.exists(index_file):
    gdown.download(f"https://drive.google.com/uc?id={INDEX_FILE_ID}", index_file, quiet=False)

if not os.path.exists(csv_file):
    gdown.download(f"https://drive.google.com/uc?id={CSV_FILE_ID}", csv_file, quiet=False)

# ==========================================
# 3. 데이터와 벡터 인덱스 로드
# ==========================================
@st.cache_resource
def load_resources():
    index = faiss.read_index(index_file)
    df = pd.read_csv(csv_file)
    return index, df

index, df = load_resources()

# ==========================================
# 4. 검색 + GPT 2단계 처리
# ==========================================
def two_pass_rag(query, top_k=20):
    # Step 1: 질문 벡터화
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    emb = np.array(emb).astype("float32").reshape(1, -1)

    # Step 2: 1차 검색
    D, I = index.search(emb, top_k)
    rows = df.iloc[I[0]]

    # Step 3: 1차 요약 (맥락 정리)
    context = "\n\n".join(
        f"- {r.get('title','')}\n엔티티:{r.get('entities','')}\n이벤트:{r.get('events','')}"
        for _, r in rows.iterrows()
    )

    summary_prompt = f"""
다음은 검색된 뉴스 기사 20개의 핵심 내용입니다.
이 자료를 기반으로 **주요 사건, 핵심 인물, 배경 흐름**을 압축해 500자 이내로 요약하세요.
연도·사람·사건을 빠짐없이 남기고, 중복은 합쳐 주세요.

검색 문서:
{context}
"""
    summary = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=600,
        messages=[{"role":"user","content":summary_prompt}]
    ).choices[0].message.content

    # Step 4: 최종 답변
    final_prompt = f"""
당신은 '슬로우레터' 스타일의 심층 분석 어시스턴트입니다.

질문: {query}

아래는 질문과 관련된 핵심 요약입니다:
{summary}

# 작성 규칙
- 첫 줄에 핵심 한 문장 요약
- 이어서 5~7개의 불릿 포인트
- 발언이나 주장에는 가능하면 (인물·기관, 날짜 등) 출처를 괄호로 명시하세요
- 최신 기사일수록 비중을 높이세요
- 건조하고 간결한 문체, 인용은 원문 그대로 사용
"""

    final_answer = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=800,
        messages=[{"role":"user","content":final_prompt}]
    ).choices[0].message.content

    return final_answer

# ==========================================
# 5. Streamlit UI
# ==========================================
st.title("Slow News Insight Bot. (2단계 RAG)")

query = st.text_area("질문을 입력하세요.")

if st.button("실행"):
    if query.strip():
        with st.spinner("검색 중..."):
            answer = two_pass_rag(query)
            st.write("### 답변")
            st.write(answer)
