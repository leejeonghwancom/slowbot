import streamlit as st
import os
import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
# import gdown # Google Drive에서 다운로드하는 로직은 더 이상 필요하지 않으므로 주석 처리 또는 제거

# ==========================================
# 0. 설정 (환경에 맞게 수정 필요)
# ==========================================
# 이 경로는 FAISS 벡터 파일과 CSV 파일이 저장된 드라이브의 'slowproject' 폴더 경로와 일치해야 합니다.
# Colab 환경이 아닌 실제 서버 배포 시에는 이 경로를 서버 파일 시스템에 맞게 조정해야 합니다.
BASE_DIR = "/content/drive/MyDrive/slowproject" # 예시 경로

# FAISS 최신 버전이 저장되는 고정 디렉토리 경로 (백엔드 시스템에서 생성)
FAISS_LATEST_DIR = os.path.join(BASE_DIR, "faiss_vectordb_latest")
# 최신 CSV 파일 경로 (백엔드 시스템에서 생성되는 덮어쓰기 파일)
CSV_LATEST_FILE = os.path.join(BASE_DIR, "slowletter_with_entities_final.csv")

# ==========================================
# 1. 환경 변수 (OpenAI API KEY)
# ==========================================
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    st.error("OpenAI API Key가 없습니다. Streamlit secrets 또는 환경 변수로 OPENAI_API_KEY를 설정하세요.")
    st.stop() # API 키가 없으면 앱 실행 중단

client = OpenAI(api_key=API_KEY)

# ==========================================
# 2. 데이터와 벡터 인덱스 로드 (캐싱 적용)
# ==========================================
@st.cache_resource # 캐시 데코레이터: 앱 재실행 시 리소스 재로드 방지
def load_resources():
    """
    지정된 경로에서 FAISS 인덱스, 문서, 메타데이터 및 전체 CSV DataFrame을 로드합니다.
    """
    st.info(f"FAISS 인덱스 및 CSV 파일 로드 중... (경로: {FAISS_LATEST_DIR})")

    # FAISS 파일 경로 구성
    index_path = os.path.join(FAISS_LATEST_DIR, "faiss_index.bin")
    docs_path = os.path.join(FAISS_LATEST_DIR, "documents.pkl")
    metadatas_path = os.path.join(FAISS_LATEST_DIR, "metadata.pkl")

    # 파일 존재 여부 확인
    if not all(os.path.exists(f) for f in [index_path, docs_path, metadatas_path]):
        st.error(f"FAISS 파일을 찾을 수 없습니다. 경로를 확인해주세요: {FAISS_LATEST_DIR}")
        st.info("먼저 백엔드 시스템('filename_based_incremental.py')을 실행하여 'faiss_vectordb_latest' 디렉토리를 생성해야 합니다.")
        st.stop() # 파일이 없으면 앱 실행 중단

    try:
        faiss_index = faiss.read_index(index_path)
        with open(docs_path, 'rb') as f:
            faiss_documents = pickle.load(f)
        with open(metadatas_path, 'rb') as f:
            faiss_metadatas = pickle.load(f)
        
        # 쿼리 임베딩에 사용될 SentenceTransformer 모델 로드 (FAISS 인덱스 생성 시 사용된 모델과 동일해야 함)
        embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    except Exception as e:
        st.error(f"FAISS 인덱스 또는 관련 파일 로드 중 오류 발생: {e}")
        st.stop()

    # 전체 CSV DataFrame 로드
    if not os.path.exists(CSV_LATEST_FILE):
        st.error(f"CSV 파일을 찾을 수 없습니다. 경로를 확인해주세요: {CSV_LATEST_FILE}")
        st.info("백엔드 시스템이 'slowletter_with_entities_final.csv' 파일을 생성했는지 확인해주세요.")
        st.stop() # 파일이 없으면 앱 실행 중단
    
    try:
        df_full = pd.read_csv(CSV_LATEST_FILE)
        # 'section_id' 컬럼이 문자열 타입인지 확인하여 검색 결과 매칭 시 오류 방지
        if 'section_id' in df_full.columns:
            df_full['section_id'] = df_full['section_id'].astype(str)
        else:
            st.warning("경고: 로드된 CSV에 'section_id' 컬럼이 없습니다. FAISS 검색 결과를 CSV 데이터와 정확히 매칭하기 어려울 수 있습니다.")
            st.warning("백엔드 시스템을 최신 버전으로 업데이트하고 데이터를 재생성해주세요.")

    except Exception as e:
        st.error(f"CSV 파일 로드 중 오류 발생: {e}")
        st.stop()

    st.success(f"모든 데이터 리소스 로드 완료! (총 {faiss_index.ntotal:,} 벡터)")
    return faiss_index, faiss_documents, faiss_metadatas, df_full, embedding_model

# 전역적으로 리소스 로드 (Streamlit 캐시에 의해 한 번만 실행됨)
faiss_index, faiss_documents, faiss_metadatas, df_full, embedding_model = load_resources()


# ==========================================
# 3. 검색 + GPT 2단계 처리
# ==========================================
def two_pass_rag(query, faiss_index, faiss_documents, faiss_metadatas, df_full, embedding_model, top_k=30):
    # Step 1: 질문 벡터화 (SentenceTransformer를 사용하여 쿼리 임베딩)
    try:
        # embedding_model (SentenceTransformer)을 사용하여 쿼리 임베딩
        emb = embedding_model.encode([query], normalize_embeddings=True).astype("float32").reshape(1, -1)
    except Exception as e:
        st.error(f"쿼리 임베딩 중 오류 발생: {e}")
        return "오류: 쿼리 임베딩 실패."

    # Step 2: 1차 FAISS 검색
    D, I = faiss_index.search(emb, top_k)
    
    # FAISS 검색 결과 (인덱스)를 바탕으로 메타데이터 및 원본 데이터 추출
    retrieved_sections_data = []
    
    # 검색된 FAISS 인덱스(I[0])를 순회하며 메타데이터와 CSV의 해당 행을 찾음
    for idx in I[0]:
        if 0 <= idx < len(faiss_metadatas): # 유효한 인덱스인지 확인
            metadata = faiss_metadatas[idx]
            section_id = metadata.get('doc_id')
            
            # section_id를 사용하여 df_full에서 해당 행을 찾음
            # 'section_id' 컬럼이 문자열로 통일되었으므로 정확히 매칭 가능
            corresponding_rows = df_full[df_full['section_id'] == section_id]
            
            if not corresponding_rows.empty:
                # 첫 번째 매칭되는 행을 사용 (section_id는 고유해야 함)
                row_data = corresponding_rows.iloc[0].to_dict()
                
                # 'title'은 FAISS 메타데이터에서는 H3 제목이지만, CSV에서는 'h3_title'
                # 'content'는 FAISS 메타데이터에서는 검색용 결합 텍스트, CSV에서는 'h3_content_text'
                # 여기서는 CSV에서 온 원본 필드들을 사용하는 것이 더 정확함.
                retrieved_sections_data.append({
                    'h3_title': row_data.get('h3_title', '제목 없음'),
                    'h3_content_text': row_data.get('h3_content_text', '내용 없음'),
                    'entities': row_data.get('entities', ''),
                    'events': row_data.get('events', ''),
                    'url': row_data.get('url', ''),
                    'date': row_data.get('date', '') # 원본 날짜
                })

    if not retrieved_sections_data:
        return "관련된 정보를 찾지 못했습니다."

    # Step 3: 1차 요약 (맥락 정리)
    # 검색된 문서들의 핵심 내용으로 요약 프롬프트 생성
    context = "\n\n".join(
        f"- 제목: {r['h3_title']}\n내용: {r['h3_content_text'][:500]}...\n엔티티:{r['entities']}\n이벤트:{r['events']}\n날짜:{r['date'].split('T')[0]}"
        for r in retrieved_sections_data
    )

    summary_prompt = f"""
다음은 검색된 뉴스 기사 꼭지들의 핵심 내용입니다.
이 자료를 기반으로 **주요 사건, 핵심 인물, 배경 흐름**을 압축해 500자 이내로 요약하세요.
연도·사람·사건을 빠짐없이 남기고, 중복은 합쳐 주세요.

검색 문서:
{context}
"""
    try:
        summary = client.chat.completions.create(
            model="gpt-4o", # 더 높은 성능의 모델 사용 가능
            max_tokens=600,
            messages=[{"role":"user","content":summary_prompt}]
        ).choices[0].message.content
    except Exception as e:
        st.error(f"요약 생성 중 오류 발생: {e}")
        return "오류: 요약 생성 실패."

    # Step 4: 최종 답변 생성
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
    try:
        final_answer = client.chat.completions.create(
            model="gpt-4o", # 최종 답변 생성 모델
            max_tokens=800,
            messages=[{"role":"user","content":final_prompt}]
        ).choices[0].message.content
    except Exception as e:
        st.error(f"최종 답변 생성 중 오류 발생: {e}")
        return "오류: 최종 답변 생성 실패."

    return final_answer

# ==========================================
# 4. Streamlit UI
# ==========================================
st.title("Slow News Insight Bot (H3 꼭지 기반)") # 앱 제목 업데이트

query_text_area = st.text_area("질문을 입력하세요:", key="query_input", height=150)

if st.button("실행", key="execute_button"):
    if query_text_area.strip():
        with st.spinner("정보 검색 및 답변 생성 중입니다..."):
            # two_pass_rag 함수에 필요한 모든 리소스 전달
            answer = two_pass_rag(query_text_area, faiss_index, faiss_documents, faiss_metadatas, df_full, embedding_model)
            st.write("### 답변")
            st.write(answer)
    else:
        st.warning("질문을 입력해주세요.")

st.markdown("---")
st.caption("FAISS 검색 시스템 by Gemini AI")
