"""Chroma 向量库封装：存 chunk、按问题检索 top-k。

Chroma 是开源的「嵌入数据库」：把文字 + 向量 + 元数据 add 进去，
query 时按相似度返回 top-k。这里我们用自己的 Embedder 生成向量，
Chroma 只负责「存 + 快查 + 持久化到磁盘」。

注意 import 顺序：必须先 import config（它会设置 protobuf 兼容开关），
再 import chromadb，否则 paddle 的老生成代码会因 protobuf 版本冲突报错。
"""
import numpy as np
import jieba
from rank_bm25 import BM25Okapi

from app.config import settings
import chromadb

from app.skills.embedding import get_embedder


class ChromaStore:
    """文档 chunk 的向量库。数据持久化到本地目录，重启不丢。"""

    def __init__(self, collection_name: str = "documents"):
        # PersistentClient：本地模式，数据存磁盘（区别于 EphemeralClient 的内存模式）
        self._client = chromadb.PersistentClient(path=settings.chroma_dir)
        # collection 类似一张表；get_or_create 重复初始化不报错
        self._collection = self._client.get_or_create_collection(name=collection_name)
        self._embedder = get_embedder()
        # 混合检索所需的内存侧数据，和 Chroma 里的 chunk 一一对应：
        # 稠密检索走 Chroma；稀疏检索（BM25）需要原始文本和向量，所以内存里留一份。
        self._docs: list[str] = []                      # chunk 原文（BM25 分词用）
        self._sources: list[str] = []                   # 每个 chunk 的来源文件名
        self._embeddings: np.ndarray | None = None      # 稠密向量（融合时直接算相似度）
        self._bm25: BM25Okapi | None = None             # 稀疏索引

    def reset(self) -> None:
        """清空并重建（demo 每次从头索引时用）。"""
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(name=name)
        # 内存侧的稀疏/稠密索引也一并清空，保持和 Chroma 同步
        self._docs = []
        self._sources = []
        self._embeddings = None
        self._bm25 = None

    def add_chunks(self, chunks: list[str], sources: list[str]) -> None:
        """把一批 chunk 向量化后入库，附带来源文件名。"""
        if not chunks:
            return
        embeddings = self._embedder.encode(chunks)  # 形状 (n, dim)
        self._collection.add(
            ids=[f"{src}#{i}" for i, src in enumerate(sources)],
            documents=chunks,
            metadatas=[{"source": src} for src in sources],
            embeddings=embeddings.tolist(),  # Chroma 要 list[list[float]]
        )
        # 同步维护混合检索需要的数据：原文 + 向量 + 重建 BM25 索引
        self._docs.extend(chunks)
        self._sources.extend(sources)
        self._embeddings = (
            embeddings if self._embeddings is None
            else np.vstack([self._embeddings, embeddings])
        )
        self._bm25 = BM25Okapi([self._tokenize(d) for d in self._docs])

    def search(self, question: str, k: int = 3) -> list[dict]:
        """检索与问题最相似的 k 个 chunk，返回 [{text, source}, ...]。"""
        if self._collection.count() == 0:
            return []  # 空库直接返回，避免 query 在空集合上报错
        q_vec = self._embedder.encode([question])[0].tolist()
        res = self._collection.query(query_embeddings=[q_vec], n_results=k)
        hits = []
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            hits.append({"text": doc, "source": meta.get("source", "?")})
        return hits

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中文分词：把一句话切成「词」列表，作为 BM25 稀疏检索的最小单元。

        jieba 对中文按词切分（如「文件整理助手」→ 文件/整理/助手），
        对英文也按空格切词，所以中英文混合文本都能用。
        """
        return jieba.lcut(text)

    def hybrid_search(self, question: str, k: int = 3, top_n: int = 20) -> list[dict]:
        """混合检索：稠密（向量相似度）+ 稀疏（BM25 关键词）→ RRF 融合取 top-k。

        为什么混合：向量检索看「语义像不像」，BM25 看「关键词像不像」，两者互补
        （比如查「转大写」，语义向量可能没排前，但「大写」这个词能硬命中）。
        融合用 RRF（Reciprocal Rank Fusion）：只认排名不认原始分数，
        score = Σ 1/(K + rank)，天然解决两路分数尺度不同的问题。
        """
        if not self._docs:
            return []
        n = len(self._docs)

        # 1. 稀疏：BM25 对每个 chunk 打关键词相关分
        bm25_scores = np.array(self._bm25.get_scores(self._tokenize(question)))

        # 2. 稠密：问题向量和每个 chunk 向量的余弦相似度（归一化后点积）
        q_vec = self._embedder.encode([question])[0]
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        e_norm = self._embeddings / (np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-9)
        dense_scores = e_norm @ q_norm  # 形状 (n,)

        # 3. 各自取 top_n 名次（分数越大越靠前 → 降序取前 top_n）
        dense_idx = np.argsort(-dense_scores)[:top_n]
        bm25_idx = np.argsort(-bm25_scores)[:top_n]

        # 4. RRF 融合：K=60 是经典常数，越小越强调第一名
        rrf: dict[int, float] = {}
        K = 60
        for rank, i in enumerate(dense_idx, start=1):
            rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (K + rank)
        for rank, i in enumerate(bm25_idx, start=1):
            rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (K + rank)

        # 5. 按融合分降序取 top-k
        top = sorted(rrf, key=rrf.get, reverse=True)[:k]
        return [{"text": self._docs[i], "source": self._sources[i]} for i in top]

    def count(self) -> int:
        """当前库里有多少条 chunk。"""
        return self._collection.count()


_store: ChromaStore | None = None


def get_store(collection_name: str = "documents") -> ChromaStore:
    """全局单例：向量库初始化较重（要加载嵌入模型），进程内只建一次。

    HTTP 接口和 Agent 工具共用这同一个库，避免重复加载模型、重复索引。
    """
    global _store
    if _store is None:
        _store = ChromaStore(collection_name=collection_name)
    return _store
