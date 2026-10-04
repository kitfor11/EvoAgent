"""RAG 评测脚本：对比「纯向量检索 vs 混合检索」，并跑三个 RAGAS 风格指标。

流程：
1. 建一个独立的向量库（collection 名 eval_documents，不污染应用的 documents 库）
2. 索引 demo_files 里的文本文件（产品手册.md、公司介绍.txt；跳过图片避免 OCR 拖慢）
3. 对每个测试问题：
   - 纯向量检索（dense）与混合检索（BM25 + 向量 + RRF）各取 top-k
   - 生成回答（走混合检索）
   - 算忠实度 / 答案相关性，并分别算两种检索的上下文相关性
4. 汇总打印 + 写 eval_result.json

真实调用 DeepSeek + 本项目的 sentence-transformers 嵌入，产出可复现的真分数。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.rag.chunker import chunk_text
from app.rag.eval_metrics import (
    answer_relevancy,
    context_relevancy,
    faithfulness,
)
from app.rag.qa import rag_answer
from app.rag.store import ChromaStore

# 测试集：前三条能答（对应两个文档），最后一条故意问文档里没有的（测「忠实度」）
QUESTIONS = [
    "RetroAgent 有哪些核心功能？",
    "RetroAgent 是什么时候、在哪里创立的？",
    "RetroAgent 的核心思想是什么？",
    "RetroAgent 支持哪些编程语言？",
]

K = 3  # 检索 top-k


def build_eval_store() -> ChromaStore:
    """建一个独立库，只索引两个文本文件（跳过图片，避免 OCR 拖慢评测）。"""
    store = ChromaStore(collection_name="eval_documents")
    chunks, sources = [], []
    for path in [ROOT / "demo_files" / "产品手册.md", ROOT / "demo_files" / "公司介绍.txt"]:
        text = path.read_text(encoding="utf-8")
        for chunk in chunk_text(text):
            chunks.append(chunk)
            sources.append(path.name)
    store.add_chunks(chunks, sources)
    return store


def main():
    print("=" * 62)
    print("RetroAgent 文档 RAG 评测：纯向量 vs 混合检索")
    print("=" * 62)

    store = build_eval_store()
    rows = []

    for q in QUESTIONS:
        dense_hits = store.search(q, k=K)          # 纯向量
        hybrid_hits = store.hybrid_search(q, k=K)  # BM25 + 向量 + RRF
        hybrid_contexts = [h["text"] for h in hybrid_hits]
        answer = rag_answer(q, store, k=K)["answer"]  # 默认走混合检索

        f = faithfulness(answer, hybrid_contexts)
        ar = answer_relevancy(q, answer)
        cr_dense = context_relevancy(q, [h["text"] for h in dense_hits])
        cr_hybrid = context_relevancy(q, hybrid_contexts)

        rows.append({
            "question": q,
            "faithfulness": f,
            "answer_relevancy": ar,
            "context_relevancy_dense": cr_dense,
            "context_relevancy_hybrid": cr_hybrid,
        })
        print(f"\nQ: {q}")
        print(f"  回答: {answer[:70]}{'...' if len(answer) > 70 else ''}")
        print(f"  上下文相关性：纯向量 {cr_dense:.2f} → 混合 {cr_hybrid:.2f}  |  忠实度 {f:.2f} | 答案相关性 {ar:.2f}")

    # 汇总
    avg = {
        "faithfulness": round(sum(r["faithfulness"] for r in rows) / len(rows), 3),
        "answer_relevancy": round(sum(r["answer_relevancy"] for r in rows) / len(rows), 3),
        "context_relevancy_dense": round(sum(r["context_relevancy_dense"] for r in rows) / len(rows), 3),
        "context_relevancy_hybrid": round(sum(r["context_relevancy_hybrid"] for r in rows) / len(rows), 3),
    }
    print("\n" + "=" * 62)
    print(f"平均上下文相关性：纯向量 {avg['context_relevancy_dense']:.2f} → 混合 {avg['context_relevancy_hybrid']:.2f}")
    print(f"平均忠实度 {avg['faithfulness']:.2f} | 平均答案相关性 {avg['answer_relevancy']:.2f}")
    print("=" * 62)

    report = {"questions": rows, "average": avg, "k": K}
    out = ROOT / "eval_result.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out.name}")


if __name__ == "__main__":
    main()
