"""问答入口：RAG 的「检索 → 增强 → 生成」三步。

1. 检索：问题向量化，从 Chroma 拿 top-k 相关 chunk；
2. 增强：把 chunk 拼成带来源编号的上下文，塞进 prompt；
3. 生成：LLM 只依据这些资料回答，并标注引用编号（不编造）。
"""
from app.llm import chat
from app.rag.store import ChromaStore


def rag_answer(question: str, store: ChromaStore, k: int = 3, hybrid: bool = True) -> dict:
    """RAG 问答，返回 {answer, sources}（接口层用，能同时拿到回答和引用来源）。

    hybrid=True 走混合检索（稠密 + BM25 稀疏 + RRF 融合），False 走纯向量检索。
    """
    hits = store.hybrid_search(question, k=k) if hybrid else store.search(question, k=k)
    sources = [h["source"] for h in hits]
    if not hits:
        return {"answer": "（没有检索到相关内容）", "sources": sources}

    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(f"[{i}] 来源《{hit['source']}》：\n{hit['text']}")
    context = "\n\n".join(parts)

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个基于检索内容的问答助手。只依据下面提供的资料回答；"
                "资料里没有的就说「资料中没有相关信息」，不要编造。"
                "回答时标注引用编号，如 [1]、[2]。"
            ),
        },
        {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
    ]
    return {"answer": chat(messages), "sources": sources}


def answer_question(question: str, store: ChromaStore, k: int = 3) -> str:
    """简化版：只返回回答文本（demo 用）。"""
    return rag_answer(question, store, k)["answer"]
