# -*- coding: utf-8 -*-
"""Geo-Graph: Streamlit LangGraph RAG chatbot using Pinecone + optional Tavily web fallback.

Deploy this file as app.py on Streamlit.
Your Pinecone index should already be populated by running index_to_pinecone.py once.

Required secrets:
- OPENAI_API_KEY
- PINECONE_API_KEY
- PINECONE_INDEX_NAME

Optional secret for online fallback:
- TAVILY_API_KEY
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, TypedDict

import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph

# ============================================================
# CONFIG
# ============================================================

DEFAULT_INDEX_NAME = "geography-kb"
DEFAULT_NAMESPACE = "default"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


class GeoGraphState(TypedDict, total=False):
    question: str
    retrieval_query: str
    corpus_docs: list[Document]
    corpus_context: str
    route: Literal["corpus", "web"]
    answer: str
    web_query: str
    web_results: list[dict[str, Any]]
    web_context: str
    answer_source: Literal["corpus", "web", "corpus_no_answer"]


# ============================================================
# SECRETS / ENVIRONMENT
# ============================================================


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None
    return value or os.getenv(name, default)


def configure_environment() -> None:
    """Set API keys into os.environ for LangChain integrations."""
    for key_name in ["OPENAI_API_KEY", "PINECONE_API_KEY", "TAVILY_API_KEY"]:
        value = get_secret(key_name)
        if value:
            os.environ[key_name] = value


# ============================================================
# CACHED CLIENTS
# ============================================================


@st.cache_resource(show_spinner=False)
def load_llm_and_vectorstore(index_name: str, namespace: str):
    """Connect to existing Pinecone index and initialise LLM."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    return llm, vectorstore


# ============================================================
# PROMPTS
# ============================================================


rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user question into a keyword-rich search query for retrieving relevant "
        "Wikipedia-style geography passages from a vector database. Keep the main entities, "
        "places, scientific terms, and topic words. Remove filler words and conversational phrasing. "
        "Do not answer the question. Return only the rewritten query.\n\n"
        "Rules:\n"
        "- For definition or 'what is' questions, keep the main concept exactly.\n"
        "- For location questions, keep the place/entity and location-related terms.\n"
        "- For 'why' questions, preserve the causal intent explicitly and include words such as "
        "causes, factors, reasons, influences, mechanisms, or effects when useful.\n"
        "- Prefer compact keyword phrases over full sentences.\n"
        "- Keep important original terms; do not replace them with vaguer synonyms.\n\n"
        "Examples:\n"
        "Question: What is geography?\n"
        "Query: geography definition study of places human environment\n\n"
        "Question: What continent is France in?\n"
        "Query: France continent Europe location geography\n\n"
        "Question: Why are deserts dry?\n"
        "Query: deserts dry causes factors low precipitation atmospheric circulation geographic barriers aridity",
    ),
    ("human", "{question}"),
])

grader_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict retrieval grader for a geography RAG chatbot. "
        "Decide whether the provided corpus context is sufficient to answer the user question.\n\n"
        "Return only one word:\n"
        "corpus = the context clearly contains the answer\n"
        "web = the context is missing, weak, unrelated, or not enough\n\n"
        "Do not explain your decision.",
    ),
    ("human", "Question:\n{question}\n\nRetrieved corpus context:\n{context}"),
])

corpus_answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are Geo-Graph, a helpful geography assistant. Answer ONLY using the provided corpus context. "
        "If the answer is not in the context, say you don't know from the indexed corpus. "
        "Provide a concise answer and cite source names when possible.",
    ),
    ("human", "Corpus context:\n{context}\n\nQuestion:\n{question}"),
])

web_query_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user question as a concise web search query. "
        "Keep entities, geography terms, locations, and dates if present. "
        "Return only the search query.",
    ),
    ("human", "{question}"),
])

web_answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are Geo-Graph, a helpful geography assistant. The indexed corpus was not enough, "
        "so you are answering from web search results. Use ONLY the provided web results. "
        "If the web results do not answer the question, say you don't know. "
        "Cite source titles or URLs when possible.",
    ),
    ("human", "Web results:\n{web_context}\n\nQuestion:\n{question}"),
])


# ============================================================
# FORMAT HELPERS
# ============================================================


def format_context(docs: list[Document]) -> str:
    lines: list[str] = []
    for doc in docs:
        src = Path(str(doc.metadata.get("source", "Unknown source"))).name
        page = doc.metadata.get("page", None)
        loc = f"{src}" + (f" p.{page}" if page is not None else "")
        lines.append(f"[Source: {loc}]\n{doc.page_content}")
    return "\n\n".join(lines)


def format_web_results(raw_result: Any) -> tuple[list[dict[str, Any]], str]:
    """Normalise Tavily output into a list and a readable context block."""
    if isinstance(raw_result, dict):
        results = raw_result.get("results", [])
        if not results and any(k in raw_result for k in ["title", "url", "content"]):
            results = [raw_result]
    elif isinstance(raw_result, list):
        results = raw_result
    else:
        results = []

    normalised: list[dict[str, Any]] = []
    blocks: list[str] = []
    for i, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        title = item.get("title", "Untitled source")
        url = item.get("url", "")
        content = item.get("content") or item.get("snippet") or item.get("raw_content") or ""
        score = item.get("score", None)
        normalised.append({"title": title, "url": url, "content": content, "score": score})
        blocks.append(f"[Web source {i}: {title}]\nURL: {url}\nContent: {content}")

    return normalised, "\n\n".join(blocks)


# ============================================================
# LANGGRAPH BUILDER
# ============================================================


def build_geo_graph(
    llm: ChatOpenAI,
    vectorstore: PineconeVectorStore,
    top_k: int,
    use_mmr: bool,
    fetch_k: int,
    lambda_mult: float,
    use_query_rewrite: bool,
    use_web_fallback: bool,
    tavily_max_results: int,
    tavily_search_depth: str,
):
    """Build a corpus-first LangGraph workflow with optional web fallback."""

    def rewrite_query_node(state: GeoGraphState) -> GeoGraphState:
        question = state["question"]
        if use_query_rewrite:
            response = llm.invoke(rewrite_prompt.format_messages(question=question))
            query = response.content.strip()
        else:
            query = question
        return {"retrieval_query": query}

    def retrieve_corpus_node(state: GeoGraphState) -> GeoGraphState:
        query = state["retrieval_query"]
        if use_mmr:
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
            )
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query)
        return {"corpus_docs": docs, "corpus_context": format_context(docs)}

    def grade_retrieval_node(state: GeoGraphState) -> GeoGraphState:
        docs = state.get("corpus_docs", [])
        context = state.get("corpus_context", "")

        if not use_web_fallback:
            return {"route": "corpus"}
        if not docs or len(context.strip()) < 100:
            return {"route": "web"}

        response = llm.invoke(
            grader_prompt.format_messages(
                question=state["question"],
                context=context[:12000],
            )
        )
        decision = response.content.strip().lower()
        return {"route": "web" if "web" in decision else "corpus"}

    def route_after_grading(state: GeoGraphState) -> str:
        return "web_search" if state.get("route") == "web" else "answer_from_corpus"

    def answer_from_corpus_node(state: GeoGraphState) -> GeoGraphState:
        response = llm.invoke(
            corpus_answer_prompt.format_messages(
                context=state.get("corpus_context", ""),
                question=state["question"],
            )
        )
        route = state.get("route", "corpus")
        return {
            "answer": response.content.strip(),
            "answer_source": "corpus" if route == "corpus" else "corpus_no_answer",
        }

    def web_search_node(state: GeoGraphState) -> GeoGraphState:
        if not os.getenv("TAVILY_API_KEY"):
            return {
                "web_query": state["question"],
                "web_results": [],
                "web_context": "",
                "answer": "I could not find the answer in the indexed corpus, and web fallback is enabled but TAVILY_API_KEY is missing.",
                "answer_source": "corpus_no_answer",
            }

        response = llm.invoke(web_query_prompt.format_messages(question=state["question"]))
        web_query = response.content.strip()
        search_tool = TavilySearch(max_results=tavily_max_results, search_depth=tavily_search_depth)
        raw_result = search_tool.invoke({"query": web_query})
        web_results, web_context = format_web_results(raw_result)
        return {"web_query": web_query, "web_results": web_results, "web_context": web_context}

    def answer_from_web_node(state: GeoGraphState) -> GeoGraphState:
        if state.get("answer"):
            return {}
        response = llm.invoke(
            web_answer_prompt.format_messages(
                web_context=state.get("web_context", ""),
                question=state["question"],
            )
        )
        return {"answer": response.content.strip(), "answer_source": "web"}

    graph = StateGraph(GeoGraphState)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve_corpus", retrieve_corpus_node)
    graph.add_node("grade_retrieval", grade_retrieval_node)
    graph.add_node("answer_from_corpus", answer_from_corpus_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("answer_from_web", answer_from_web_node)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_corpus")
    graph.add_edge("retrieve_corpus", "grade_retrieval")
    graph.add_conditional_edges(
        "grade_retrieval",
        route_after_grading,
        {
            "answer_from_corpus": "answer_from_corpus",
            "web_search": "web_search",
        },
    )
    graph.add_edge("answer_from_corpus", END)
    graph.add_edge("web_search", "answer_from_web")
    graph.add_edge("answer_from_web", END)

    return graph.compile()


def answer_question_with_graph(
    question: str,
    llm: ChatOpenAI,
    vectorstore: PineconeVectorStore,
    top_k: int,
    use_mmr: bool,
    fetch_k: int,
    lambda_mult: float,
    use_query_rewrite: bool,
    use_web_fallback: bool,
    tavily_max_results: int,
    tavily_search_depth: str,
) -> GeoGraphState:
    app_graph = build_geo_graph(
        llm=llm,
        vectorstore=vectorstore,
        top_k=top_k,
        use_mmr=use_mmr,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        use_query_rewrite=use_query_rewrite,
        use_web_fallback=use_web_fallback,
        tavily_max_results=tavily_max_results,
        tavily_search_depth=tavily_search_depth,
    )
    return app_graph.invoke({"question": question})


# ============================================================
# STREAMLIT UI
# ============================================================


st.set_page_config(page_title="Geo-Graph", page_icon="🌐", layout="wide")
configure_environment()

st.title("🌐 Geo-Graph")
st.caption(
    "LangGraph geography RAG chatbot using Pinecone first, with optional Tavily web search fallback "
    "when the indexed corpus is not enough."
)

missing = [name for name in ["OPENAI_API_KEY", "PINECONE_API_KEY"] if not os.getenv(name)]
if missing:
    st.error(
        "Missing required secrets: " + ", ".join(missing) +
        ". Add them in Streamlit Cloud → App settings → Secrets."
    )
    st.stop()

with st.sidebar:
    st.header("Geo-Graph settings")
    index_name = st.text_input("Pinecone index name", value=get_secret("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME))
    namespace = st.text_input("Pinecone namespace", value=get_secret("PINECONE_NAMESPACE", DEFAULT_NAMESPACE))

    st.subheader("Corpus retrieval")
    top_k = st.slider("TOP_K", min_value=1, max_value=20, value=8)
    use_mmr = st.toggle("Use MMR retrieval", value=True)
    fetch_k = st.slider("FETCH_K", min_value=top_k, max_value=60, value=max(20, top_k * 3))
    lambda_mult = st.slider("MMR lambda", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    use_query_rewrite = st.toggle("Rewrite query before retrieval", value=True)

    st.subheader("LangGraph web fallback")
    use_web_fallback = st.toggle("Use web search if corpus is weak", value=True)
    tavily_max_results = st.slider("Web results", min_value=1, max_value=10, value=4)
    tavily_search_depth = st.selectbox("Web search depth", options=["basic", "advanced"], index=0)

    if use_web_fallback and not os.getenv("TAVILY_API_KEY"):
        st.warning("Web fallback is on, but TAVILY_API_KEY is missing. Corpus answers still work.")

try:
    llm, vectorstore = load_llm_and_vectorstore(index_name=index_name, namespace=namespace)
except Exception as exc:
    st.error(f"Could not connect to Pinecone/OpenAI: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a geography question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Running Geo-Graph workflow..."):
            try:
                result = answer_question_with_graph(
                    question=question,
                    llm=llm,
                    vectorstore=vectorstore,
                    top_k=top_k,
                    use_mmr=use_mmr,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    use_query_rewrite=use_query_rewrite,
                    use_web_fallback=use_web_fallback,
                    tavily_max_results=tavily_max_results,
                    tavily_search_depth=tavily_search_depth,
                )

                answer = result.get("answer", "I don't know.")
                answer_source = result.get("answer_source", "corpus")

                if answer_source == "web":
                    st.info("Answer source: web search fallback")
                elif result.get("route") == "web":
                    st.warning("Corpus was weak; attempted web fallback.")
                else:
                    st.success("Answer source: indexed Pinecone corpus")

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                with st.expander("LangGraph trace"):
                    st.write("Corpus retrieval query:", result.get("retrieval_query", ""))
                    st.write("Route decision:", result.get("route", "corpus"))
                    if result.get("web_query"):
                        st.write("Web search query:", result.get("web_query"))

                with st.expander("Retrieved corpus context"):
                    docs = result.get("corpus_docs", [])
                    for i, doc in enumerate(docs, start=1):
                        source = doc.metadata.get("source", "Unknown source")
                        st.markdown(f"**Chunk {i} — Source: {source}**")
                        st.write(doc.page_content[:1200])
                        st.divider()

                if result.get("web_results"):
                    with st.expander("Web search results"):
                        for i, item in enumerate(result.get("web_results", []), start=1):
                            title = item.get("title", "Untitled source")
                            url = item.get("url", "")
                            content = item.get("content", "")
                            st.markdown(f"**Web source {i}: {title}**")
                            if url:
                                st.write(url)
                            st.write(content[:1200])
                            st.divider()
            except Exception as exc:
                st.error(f"Error while answering: {exc}")
