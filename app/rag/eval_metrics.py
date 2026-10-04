"""RAG 评测指标：按 RAGAS 的方法论自研实现（不依赖 ragas 库）。

RAGAS 是业界常用的 RAG 评测框架，核心思想是「用 LLM 当裁判 + embedding 算相似度」
把原来要人工逐条看的质量，变成可自动、可复现的分数。

这里实现三个「不需要标准答案」的指标（最诚实，只评判检索和生成本身）：
- faithfulness 忠实度：答案是不是真的基于检索资料（有没有编造）
- answer_relevancy 答案相关性：答案有没有答到点子上
- context_relevancy 上下文相关性：检索回来的片段和问题相不相关

每个指标都返回 0~1 的浮点数（越接近 1 越好）。
"""
import json
import re

from app.llm import chat
from app.skills.embedding import cosine_similarity, get_embedder


def _json_array(raw: str) -> list[str]:
    """把 LLM 的输出解析成字符串数组。容忍 markdown 代码块和多余文字。"""
    raw = raw.strip()
    # 去掉 ```json ... ``` 这类代码块围栏
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 模型偶尔会在数组前后多写一句人话，兜底取第一个 [ ... ]
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group(0))
    if isinstance(data, list):
        return [str(x) for x in data]
    return []


def faithfulness(answer: str, contexts: list[str]) -> float:
    """忠实度：答案里的事实陈述，有多大比例能被检索资料支持。

    方法（RAGAS）：
    1. 把答案拆成若干条独立的「事实陈述」（claim）
    2. 逐条问 LLM：这条陈述是否被资料支持？
    3. score = 被支持的条数 / 总条数
    """
    if not contexts or not answer.strip():
        return 0.0

    # 1. 拆陈述
    raw = chat([{
        "role": "user",
        "content": (
            "把下面这段回答拆成若干条独立的、可验证的事实陈述（claims）。\n"
            f"回答：\n{answer}\n\n"
            'Output ONLY a JSON array of strings, e.g. ["claim1", "claim2"]. '
            '如果回答里没有任何事实陈述（例如只说「资料中没有」），输出 []。'
        ),
    }])
    claims = _json_array(raw)
    if not claims:
        return 1.0  # 没有任何事实陈述 = 没有编造任何东西，视为忠实

    # 2. 逐条判断是否被资料支持
    ctx_text = "\n".join(f"- {c}" for c in contexts)
    supported = 0
    for claim in claims:
        verdict = chat([{
            "role": "user",
            "content": (
                "判断下面这条陈述是否被给出的资料支持（资料里明确提到了，才算支持）。\n"
                f"资料：\n{ctx_text}\n\n"
                f"陈述：\n{claim}\n\n"
                'Output ONLY "yes" or "no".'
            ),
        }]).strip().lower()
        if verdict.startswith("yes"):
            supported += 1
    return supported / len(claims)


def answer_relevancy(question: str, answer: str) -> float:
    """答案相关性：从答案反推它能回答哪些问题，再看和原问题像不像。

    方法（RAGAS）：
    1. 让 LLM 根据答案生成 N 个它「可能能回答」的问题
    2. 分别算这 N 个问题与原问题的向量余弦相似度
    3. score = 这些相似度的均值
    """
    if not answer.strip():
        return 0.0

    raw = chat([{
        "role": "user",
        "content": (
            "根据下面的回答，生成 3 个这个回答可能能回答的问题。\n"
            f"回答：\n{answer}\n\n"
            'Output ONLY a JSON array of 3 strings.'
        ),
    }])
    generated = _json_array(raw)
    if not generated:
        return 0.0

    embedder = get_embedder()
    vecs = embedder.encode([question] + generated)
    q_vec = vecs[0]
    sims = [cosine_similarity(q_vec, vecs[i + 1]) for i in range(len(generated))]
    score = sum(sims) / len(sims)
    return max(0.0, min(1.0, score))  # 余弦相似度理论上可能略超 1，夹到 [0,1]


def context_relevancy(question: str, contexts: list[str]) -> float:
    """上下文相关性：检索回来的片段里，有多大比例的句子和问题相关。

    方法（RAGAS）：
    1. 对每条检索片段，让 LLM 提取出与问题相关的句子
    2. score = 所有片段里相关句子总数 / 所有片段句子总数
    """
    if not contexts:
        return 0.0

    total_sentences = 0
    relevant_sentences = 0
    for ctx in contexts:
        # 粗切句：按中英文句号/问号/感叹号分句（足够用于统计比例）
        sents = [s.strip() for s in re.split(r"[。！？.!?]", ctx) if s.strip()]
        total_sentences += len(sents)
        if not sents:
            continue

        raw = chat([{
            "role": "user",
            "content": (
                "从下面资料中，提取与问题相关的句子（原文照抄，不要改写）。\n"
                f"问题：{question}\n"
                f"资料：\n{ctx}\n\n"
                'Output ONLY a JSON array of relevant sentences. 没有相关句子就输出 []。'
            ),
        }])
        relevant_sentences += len(_json_array(raw))

    if total_sentences == 0:
        return 0.0
    return min(1.0, relevant_sentences / total_sentences)
