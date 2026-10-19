import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlsplit

import requests
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

PDF_PATH = Path(__file__).parent / "homabot_manual.pdf"
EMBED_MODEL = "text-embedding-3-small"


import os


def get_openai_api_key() -> str:
    key = ""
    try:
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            key = str(st.secrets["OPENAI_API_KEY"]).strip()
    except Exception:
        pass
    if not key:
        key = os.getenv("OPENAI_API_KEY", "").strip()
    return key


def get_serpapi_key() -> str:
    key = ""
    try:
        if hasattr(st, "secrets"):
            key = str(st.secrets.get("SERPAPI_KEY") or st.secrets.get("SERPAPI_API_KEY") or "").strip()
    except Exception:
        pass
    if not key:
        key = (os.getenv("SERPAPI_KEY") or os.getenv("SERPAPI_API_KEY") or "").strip()
    return key


def get_openai_model() -> str:
    model = "gpt-4o-mini"
    try:
        if hasattr(st, "secrets") and "OPENAI_MODEL" in st.secrets:
            model = str(st.secrets["OPENAI_MODEL"]).strip()
    except Exception:
        pass
    if not model:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    return model or "gpt-4o-mini"


def is_valid_key(key: str) -> bool:
    if not key or not isinstance(key, str):
        return False
    k = key.strip().strip('"').strip("'")
    return len(k) > 10 and not k.startswith(("운영팀에서_", "실제 "))


def openai_error_message(exc: Exception) -> str:
    if isinstance(exc, AuthenticationError):
        return "OpenAI 인증 오류: API 키를 확인하세요."
    if isinstance(exc, PermissionDeniedError):
        return "OpenAI 권한 오류: 프로젝트와 모델 사용 권한을 확인하세요."
    if isinstance(exc, RateLimitError):
        return "OpenAI 429 오류: 요청 한도 또는 잔액을 확인하세요."
    if isinstance(exc, APITimeoutError):
        return "OpenAI 응답 시간이 초과되었습니다."
    if isinstance(exc, APIConnectionError):
        return "OpenAI 연결 오류: 네트워크 상태를 확인하세요."
    if isinstance(exc, (BadRequestError, NotFoundError)):
        return "OpenAI 요청 오류: 모델 이름 및 설정을 확인하세요."
    return f"OpenAI API 오류({type(exc).__name__})."


# ==========================================
# 상태(State) 정의
# ==========================================
class SearchResult(TypedDict):
    title: str
    snippet: str
    link: str
    date: str


class ManualDoc(TypedDict):
    page: int
    content: str


class QAState(TypedDict, total=False):
    question: str
    user_mode: str  # "자동 선택", "매뉴얼 검색", "웹 검색", "매뉴얼과 웹 검색 비교"
    route: str  # "manual", "web", "compare"
    route_reason: str
    openai_key: str
    serp_key: str
    manual_context: list[ManualDoc]
    search_results: list[SearchResult]
    searched_at: str
    answer: str
    error: str


# ==========================================
# PDF RAG 검색 인덱스 준비
# ==========================================
@st.cache_resource(show_spinner="HomaBot 매뉴얼 인덱스 준비 중...")
def get_vector_retriever(api_key: str):
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"{PDF_PATH.name} 파일을 찾을 수 없습니다.")
    pages = PyPDFLoader(str(PDF_PATH)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    splits = splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=api_key)
    vectordb = FAISS.from_documents(splits, embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 3})


# ==========================================
# 노드 1: 질문 경로 라우팅 (규칙 우선 + LLM Fallback)
# ==========================================
MANUAL_KEYWORDS = [
    "설정", "버튼", "청소", "배터리", "wi-fi", "와이파이", "수면", "모드", "펌웨어", "업데이트",
    "브러시", "충전", "사양", "스펙", "hx-1", "hx-2", "hx-pro", "sos", "긴급", "매뉴얼", "호마봇", "homabot",
]
WEB_KEYWORDS = [
    "오늘", "현재", "최근", "실시간", "날씨", "주가", "뉴스", "환율", "정책", "가격", "얼마",
]
COMPARE_KEYWORDS = [
    "비교", "차이", "어느 모델", "어느쪽", "맞나요", "다르", "일치", "어떤게 맞아",
]


def route_question(state: QAState) -> QAState:
    mode = state.get("user_mode", "자동 선택")
    question = state.get("question", "").strip()
    q_lower = question.lower()

    if mode == "매뉴얼 검색":
        return {"route": "manual", "route_reason": "사용자 수동 선택 (매뉴얼)"}
    if mode == "웹 검색":
        return {"route": "web", "route_reason": "사용자 수동 선택 (웹 검색)"}
    if mode == "매뉴얼과 웹 검색 비교":
        return {"route": "compare", "route_reason": "사용자 수동 선택 (비교 검색)"}

    # 자동 선택: 규칙 기반 1차 검사
    has_manual = any(k in q_lower for k in MANUAL_KEYWORDS)
    has_web = any(k in q_lower for k in WEB_KEYWORDS)
    has_compare = any(k in q_lower for k in COMPARE_KEYWORDS)

    if has_compare and (has_manual or "공기질" in q_lower or "모델" in q_lower):
        return {"route": "compare", "route_reason": "규칙 분류: 자료 비교/확인 질의"}
    if has_web and not has_manual:
        return {"route": "web", "route_reason": "규칙 분류: 최신/외부 정보 질의"}
    if has_manual and not has_web:
        return {"route": "manual", "route_reason": "규칙 분류: 제품 사양/매뉴얼 질의"}

    # 규칙으로 판단 불확실할 때 LLM 1-토큰 분류
    openai_key = get_openai_api_key()
    if is_valid_key(openai_key):
        try:
            client = OpenAI(api_key=openai_key, timeout=10.0, max_retries=1)
            resp = client.chat.completions.create(
                model=get_openai_model(),
                temperature=0,
                max_tokens=10,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "질문을 [manual, web, compare] 중 하나로만 분류하세요.\n"
                            "- manual: 로봇(HomaBot) 기능, 버튼, 와이파이, 청소, 사용법, 스펙 등\n"
                            "- web: 실시간 날씨, 시사, 일반 상식, 최신 뉴스, 외부 정보 등\n"
                            "- compare: 매뉴얼과 최신 외부 정보의 차이점이나 비교가 필요한 경우\n"
                            "단어 하나만 출력하세요."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
            )
            route_pred = (resp.choices[0].message.content or "").strip().lower()
            if "compare" in route_pred:
                return {"route": "compare", "route_reason": "LLM 분류: 비교 질의"}
            if "web" in route_pred:
                return {"route": "web", "route_reason": "LLM 분류: 웹 정보 질의"}
            return {"route": "manual", "route_reason": "LLM 분류: 제품 매뉴얼 질의"}
        except Exception:
            pass

    # 최종 fallback: 안전하게 매뉴얼 우선
    return {"route": "manual", "route_reason": "분류 기본값 (매뉴얼 우선)"}


# ==========================================
# 노드 2: 매뉴얼 검색 노드
# ==========================================
def manual_search_node(state: QAState) -> QAState:
    question = state.get("question", "").strip()
    openai_key = state.get("openai_key") or get_openai_api_key()
    if not is_valid_key(openai_key):
        return {"error": "OPENAI_API_KEY가 누락되었습니다."}
    try:
        retriever = get_vector_retriever(openai_key)
        docs = retriever.invoke(question)
        manual_context: list[ManualDoc] = [
            {"page": int(d.metadata.get("page", 0)) + 1, "content": d.page_content}
            for d in docs
        ]
        return {"manual_context": manual_context}
    except Exception as exc:
        return {"manual_context": [], "error": f"매뉴얼 검색 오류: {exc}"}


# ==========================================
# 노드 3: 웹 검색 노드 (SerpAPI)
# ==========================================
def web_search_node(state: QAState) -> QAState:
    question = state.get("question", "").strip()
    searched_at = datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")
    serp_key = state.get("serp_key") or get_serpapi_key()
    if not is_valid_key(serp_key):
        return {
            "search_results": [],
            "searched_at": searched_at,
            "error": "SERPAPI_KEY가 설정되지 않았습니다 (키 길이 또는 형식 확인 필요).",
        }

    params = {
        "q": question,
        "api_key": serp_key,
        "engine": "google",
        "hl": "ko",
        "gl": "kr",
        "output": "json",
    }
    try:
        res = requests.get("https://serpapi.com/search", params=params, timeout=(5, 20))
        res.raise_for_status()
        data = res.json()
        organic = data.get("organic_results", []) or []
        results: list[SearchResult] = []
        seen = set()
        for item in organic:
            link = str(item.get("link", "")).strip()
            title = str(item.get("title", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            if not link or link in seen or not (title or snippet):
                continue
            parsed = urlsplit(link)
            if parsed.scheme not in {"http", "https"}:
                continue
            results.append({
                "title": title[:300] or "제목 없음",
                "snippet": snippet[:1500],
                "link": link,
                "date": str(item.get("date", "")).strip(),
            })
            seen.add(link)
            if len(results) >= 5:
                break
        return {"search_results": results, "searched_at": searched_at}
    except Exception as exc:
        return {
            "search_results": [],
            "searched_at": searched_at,
            "error": f"웹 검색 오류: {exc}",
        }


# ==========================================
# 노드 4: 통합 답변 생성 노드
# ==========================================
SYSTEM_PROMPT = """당신은 HomaBot 제품 안내 및 웹 정보 통합 상담 어시스턴트입니다.
답변 작성 시 다음 핵심 규칙을 반드시 지키세요:

1. [매뉴얼 검색 자료 및 내부 모순/충돌 처리]:
   - 반드시 제공된 [매뉴얼 자료] 내용만을 바탕으로 답변하세요.
   - 매뉴얼에 없는 내용(예: 보증 기간, 표기되지 않은 부품 가격 등)은 절대로 지어내지 말고, "매뉴얼에서 확인할 수 없습니다"라고 분명하게 안내하세요.
   - [중요: 매뉴얼 내부 내용 충돌 및 한계 명시]
     * 본 문서는 실습용 가상 자료이며 출시 연도가 2029~2031년 등 미래로 표시되어 있으므로 실제 현실의 확정 사실로 단정하지 마세요.
     * 공기질 모니터링/관리 관련 질의처럼 매뉴얼 내부 표기와 본문이 상충하는 경우:
       - 1.2절 모델별 특징 비교표: "HX-PRO" 모델에만 '공기질 관리'가 명시되어 있음.
       - 3.2절 상세 기능 본문: "3.2 HX-2 공기질 모니터링"으로 명시되어 있음.
       -> 이와 같이 매뉴얼 내부에 서로 모순·충돌하는 내용이 있음을 명확히 짚어주고, 두 내용을 모두 제시하여 설명하세요. 임의로 하나만 맞다고 결론짓지 마세요.

2. [웹 검색 자료]:
   - 제공된 제목과 요약(snippet)에 근거해서만 답하고, 본문을 열람했다고 표현하지 마세요.
   - 가상 제품(HomaBot)에 대한 일반 웹 검색 결과가 무관한 내용(노래 제목, 일상 글 등)뿐이라면, "웹 검색에서는 HomaBot 관련 신뢰할 만한 정보를 찾을 수 없습니다"라고 명확히 밝히세요.
   - 근거 문장에는 [1], [2] 번호를 달고, 본문에 가짜 URL을 직접 작성하지 마세요.

3. [자료 한계 및 정직성]:
   - 불확실하거나 근거가 부족하면 추측하지 말고 한계를 솔직하게 안내하세요.
"""


def generate_answer_node(state: QAState) -> QAState:
    error = state.get("error", "")
    route = state.get("route", "manual")
    question = state.get("question", "")
    manual_docs = state.get("manual_context", [])
    web_results = state.get("search_results", [])
    openai_key = state.get("openai_key") or get_openai_api_key()

    # 전체 실패 검사
    if route == "web" and error:
        return {"answer": f"[검색 실패] {error}"}
    if route == "web" and not web_results:
        return {"answer": "사용할 수 있는 웹 검색 결과가 없습니다. 검색어를 구체적으로 입력하세요."}
    if route == "manual" and error:
        return {"answer": f"[매뉴얼 오류] {error}"}

    context_payload = {
        "question": question,
        "selected_route": route,
        "route_reason": state.get("route_reason", ""),
        "manual_context": [
            {"page": d["page"], "text": d["content"]} for d in manual_docs
        ],
        "web_results": [
            {"id": i, **item} for i, item in enumerate(web_results, start=1)
        ],
        "searched_at": state.get("searched_at", ""),
    }

    try:
        client = OpenAI(api_key=openai_key, timeout=60.0, max_retries=2)
        resp = client.chat.completions.create(
            model=get_openai_model(),
            temperature=0.2,
            max_tokens=1200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(context_payload, ensure_ascii=False)},
            ],
        )
        answer = (resp.choices[0].message.content or "").strip()
    except OpenAIError as exc:
        return {"answer": f"[답변 생성 실패] {openai_error_message(exc)}"}

    # 출처 및 근거 부착
    citations = []
    if manual_docs and route in {"manual", "compare"}:
        pages = sorted({d["page"] for d in manual_docs})
        if "매뉴얼에서 확인할 수 없습니다" not in answer:
            citations.append(f"📄 **참조 매뉴얼 페이지:** {', '.join(f'{p}페이지' for p in pages)}")

    if web_results and route in {"web", "compare"}:
        source_lines = [
            f"- [{i}] **{r['title']}** ({r['date'] or '작성일 미제공'})  \n  {r['link']}"
            for i, r in enumerate(web_results, start=1)
        ]
        citations.append("**🌐 웹 검색 출처 목록 (웹페이지 본문 미열람):**\n" + "\n".join(source_lines))

    if citations:
        answer += "\n\n---\n" + "\n\n".join(citations)

    return {"answer": answer}


# ==========================================
# LangGraph 그래프 빌드
# ==========================================
def build_advanced_graph():
    wf = StateGraph(QAState)
    wf.add_node("route_question", route_question)
    wf.add_node("manual_search", manual_search_node)
    wf.add_node("web_search", web_search_node)
    wf.add_node("generate_answer", generate_answer_node)

    wf.add_edge(START, "route_question")

    # 조건부 라우팅 분기
    def route_decision(state: QAState) -> str:
        r = state.get("route", "manual")
        return "compare" if r == "compare" else r

    wf.add_conditional_edges(
        "route_question",
        route_decision,
        {
            "manual": "manual_search",
            "web": "web_search",
            "compare": "manual_search",
        },
    )

    wf.add_edge("manual_search", "generate_answer")
    wf.add_edge("web_search", "generate_answer")

    # compare 분기일 경우: manual_search 후 web_search로 이어짐
    # (단순하고 직관적인 선형 파이프라인 처리를 위해 generate_answer 이전에 상태 체크)
    # LangGraph 조건부 엣지: manual_search 후 compare면 web_search로
    def post_manual_decision(state: QAState) -> str:
        return "web_search" if state.get("route") == "compare" else "generate_answer"

    # 기존 direct edge 대신 교체
    wf.add_conditional_edges(
        "manual_search",
        post_manual_decision,
        {
            "web_search": "web_search",
            "generate_answer": "generate_answer",
        },
    )

    wf.add_edge("generate_answer", END)
    return wf.compile()


@st.cache_resource
def get_compiled_graph():
    return build_advanced_graph()


graph_app = get_compiled_graph()


# ==========================================
# Streamlit 화면 구성 (미니멀 UI)
# ==========================================
st.set_page_config(page_title="HomaBot 통합 지식 상담 챗봇", page_icon="🤖")
st.title("HomaBot 통합 지식 상담 챗봇")
st.caption("PDF 매뉴얼 RAG와 실시간 웹 검색(SerpAPI)을 질문에 따라 유연하게 통합 처리합니다.")

openai_status = "✅ 설정됨" if is_valid_key(get_openai_api_key()) else "❌ 미설정"
serp_status = "✅ 설정됨" if is_valid_key(get_serpapi_key()) else "❌ 미설정"

with st.sidebar:
    st.subheader("🔑 API 키 설정 상태")
    st.write(f"- OpenAI: {openai_status}")
    st.write(f"- SerpAPI: {serp_status}")
    if not is_valid_key(get_openai_api_key()) or not is_valid_key(get_serpapi_key()):
        st.info("Streamlit 대시보드의 Settings -> Secrets에 키가 올바르게 입력되었는지 확인하세요.")

if not is_valid_key(get_openai_api_key()) or not is_valid_key(get_serpapi_key()):
    st.warning(f"⚠️ API 키 설정 상태를 확인하세요 (OpenAI: {openai_status}, SerpAPI: {serp_status})")

col1, col2 = st.columns([4, 1])
with col1:
    user_mode = st.radio(
        "처리 방식",
        ["자동 선택", "매뉴얼 검색", "웹 검색", "매뉴얼과 웹 검색 비교"],
        horizontal=True,
    )
with col2:
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.advanced_messages = []
        st.rerun()

if "advanced_messages" not in st.session_state:
    st.session_state.advanced_messages = []

for msg in st.session_state.advanced_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

raw_input = st.chat_input("질문을 입력하세요 (예: Wi-Fi 설정 방법, 공기질 모니터링 모델, 오늘 날씨)")

if raw_input is not None:
    question = raw_input.strip()
    if not question:
        st.info("질문을 입력해 주세요.")
    else:
        st.session_state.advanced_messages.append({
            "role": "user",
            "content": f"**[{user_mode}]** {question}",
        })
        with st.chat_message("user"):
            st.markdown(f"**[{user_mode}]** {question}")

        with st.chat_message("assistant"):
            with st.status("답변을 처리하는 중...", expanded=False) as status:
                st.write("1. 질문 분석 및 검색 경로 결정...")
                initial_state: QAState = {
                    "question": question,
                    "user_mode": user_mode,
                    "openai_key": get_openai_api_key(),
                    "serp_key": get_serpapi_key(),
                    "manual_context": [],
                    "search_results": [],
                    "searched_at": "",
                    "answer": "",
                    "error": "",
                }
                final_state = graph_app.invoke(initial_state)
                route_used = final_state.get("route", "")
                route_reason = final_state.get("route_reason", "")
                st.write(f"2. 실행 경로: **{route_used}** ({route_reason})")
                st.write("3. 답변 및 근거 정리 완료")
                status.update(label=f"완료 (경로: {route_used})", state="complete")

            full_answer = (
                f"> **[처리 경로: {final_state.get('route')}]** {final_state.get('route_reason')}\n\n"
                + final_state.get("answer", "")
            )
            st.markdown(full_answer)
            st.session_state.advanced_messages.append({
                "role": "assistant",
                "content": full_answer,
            })
