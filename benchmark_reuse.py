"""评测脚本：测「技能复用 vs 从零求解」的真实速度与成功率。

思路：同任务跑两轮
- 冷启动（空技能库）：Agent 从零求解，并沉淀成技能 —— 这是「未复用」基线
- 热启动（同一技能库已有沉淀）：Agent 检索命中后复用 —— 这是「复用」

度量：
- 耗时（墙钟时间，直接 time.perf_counter）
- LLM 调用次数（monkeypatch 包装 chat，统计真实调用）
- method（solved / reused_skill / failed）
- 复用命中率、成功率、提速比

用临时 JSON 存储，不碰 Redis、不污染真实 skills.json / memories.json。
"""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import app.llm as llm
import app.agent.react_loop as react_loop
import app.agent.reflect as reflect_mod
from app.agent.agent import solve_task
from app.memory.store import JsonMemoryStore
from app.skills.embedding import get_embedder
from app.skills.store import JsonSkillStore

# ---- 统计 LLM 调用：包装 chat，计数并透传 ----
_llm_calls = {"n": 0}
_orig_chat = llm.chat


def counting_chat(messages, **kwargs):
    _llm_calls["n"] += 1
    return _orig_chat(messages, **kwargs)


# react_loop / reflect 都是 `from app.llm import chat`，导入时已绑死名字，
# 所以要同时替换这三处引用，才能把 solve_task 内部所有调用都统计进去。
llm.chat = counting_chat
react_loop.chat = counting_chat
reflect_mod.chat = counting_chat

# 任务集：覆盖工具集的几类纯文本操作，都是「第一次能解、第二次能复用」的
TASKS = [
    "Reverse the text 'hello world'.",
    "Count the words in 'the quick brown fox jumps'.",
    "Convert the text 'hello agent' to uppercase.",
    "Convert the text 'HELLO AGENT' to lowercase.",
    "List all files in the current directory.",
]


def run_round(skill_store, memory_store):
    """跑一轮：对每个任务 solve_task，记录耗时/调用数/方法。"""
    results = []
    for task in TASKS:
        _llm_calls["n"] = 0
        t0 = time.perf_counter()
        try:
            out = solve_task(task, skill_store, memory_store)
            method = out.get("method")
            trace = out.get("trace", [])
            err = None
        except Exception as ex:  # 网络抖动/JSON 解析失败等，单任务失败不崩整体
            method = "error"
            trace = []
            err = repr(ex)
        dt = time.perf_counter() - t0
        results.append({
            "task": task,
            "method": method,
            "time_s": round(dt, 2),
            "llm_calls": _llm_calls["n"],
            "trace_steps": len(trace),
            "error": err,
        })
    return results


def summarize(results, label):
    """汇总一轮：平均耗时、平均调用数、命中率、成功率。"""
    n = len(results)
    times = [r["time_s"] for r in results]
    calls = [r["llm_calls"] for r in results]
    ok = [r for r in results if r["method"] in ("solved", "reused_skill")]
    hit = [r for r in results if r["method"] == "reused_skill"]
    print(f"\n===== {label} =====")
    for r in results:
        err = f"  ERROR={r['error']}" if r["error"] else ""
        print(f"  [{r['method']:<14}] {r['time_s']:>5.2f}s  calls={r['llm_calls']:>2}  {r['task']}{err}")
    print(f"  avg time   : {sum(times)/n:.2f}s")
    print(f"  avg calls  : {sum(calls)/n:.2f}")
    print(f"  success    : {len(ok)}/{n}")
    print(f"  reuse hit  : {len(hit)}/{n}")
    return {
        "avg_time_s": round(sum(times) / n, 2),
        "avg_llm_calls": round(sum(calls) / n, 2),
        "success_rate": round(len(ok) / n, 3),
        "reuse_hit_rate": round(len(hit) / n, 3),
    }


def main():
    get_embedder()  # 提前加载嵌入模型，避免计入首任务耗时
    print("=" * 60)
    print("RetroAgent 技能复用评测（真实调用 DeepSeek）")
    print("=" * 60)

    tmp = Path(tempfile.mkdtemp(prefix="retroagent_bench_"))
    skill_store = JsonSkillStore(str(tmp / "skills.json"))
    memory_store = JsonMemoryStore(str(tmp / "memories.json"))

    cold = run_round(skill_store, memory_store)   # 空库：从零求解
    cold_sum = summarize(cold, "冷启动（未复用）")

    warm = run_round(skill_store, memory_store)   # 已沉淀：检索复用
    warm_sum = summarize(warm, "热启动（复用）")

    speedup = cold_sum["avg_time_s"] / warm_sum["avg_time_s"]
    call_reduce = (1 - warm_sum["avg_llm_calls"] / cold_sum["avg_llm_calls"]) * 100

    print("\n===== 结论 =====")
    print(f"  速度提升      : 复用比未复用快 {speedup:.2f}x（{cold_sum['avg_time_s']}s -> {warm_sum['avg_time_s']}s）")
    print(f"  LLM 调用减少  : {call_reduce:.0f}%（{cold_sum['avg_llm_calls']} -> {warm_sum['avg_llm_calls']} 次/任务）")
    print(f"  复用命中率    : {warm_sum['reuse_hit_rate']*100:.0f}%")
    print(f"  冷启动成功率  : {cold_sum['success_rate']*100:.0f}%")
    print(f"  热启动成功率  : {warm_sum['success_rate']*100:.0f}%")

    report = {
        "tasks": TASKS,
        "cold": cold,
        "warm": warm,
        "summary": {
            "speedup_x": round(speedup, 2),
            "llm_call_reduction_pct": round(call_reduce, 0),
            **cold_sum,
            "cold_success_rate": cold_sum["success_rate"],
            "warm_success_rate": warm_sum["success_rate"],
            "reuse_hit_rate": warm_sum["reuse_hit_rate"],
        },
    }
    out = ROOT / "benchmark_result.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out.name}")


if __name__ == "__main__":
    main()
