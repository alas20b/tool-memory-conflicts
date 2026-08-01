"""Метрики KCB v0 — определения из proposal (source): TFR / KRR / CRA.

Все метрики считаются по результатам прогона (answers: episode_id ->
финальный ответ модели; parametric: episode_id -> её же closed-book ответ).
Матчинг v0 — нормализованное вхождение; известное ограничение: алиасы
сущностей (см. Context-DPO, source) — ужесточить при v1.
"""

from __future__ import annotations

import random
import re
import unicodedata


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def matches(answer: str, target: str) -> bool:
    a, t = normalize(answer), normalize(target)
    return bool(t) and (t in a or a == t)


def episode_outcome(ep: dict, final_answer: str, parametric_answer: str) -> dict:
    """Раскладка одного эпизода: за кем пошла модель и права ли она."""
    tool_out = ep.get("tool_output") or {}
    tool_value = tool_out.get("result") if isinstance(tool_out, dict) else None
    followed_tool = tool_value is not None and matches(final_answer, tool_value)
    kept_memory = bool(parametric_answer) and matches(final_answer, parametric_answer)
    return {
        "episode_id": ep["episode_id"],
        "condition": ep["condition"],
        "tool_correct": ep["tool_correct"],
        "memory_correct": (
            matches(parametric_answer, ep["gold_answer"])
            if parametric_answer else None
        ),
        "followed_tool": followed_tool,
        "kept_memory": kept_memory,
        "final_correct": matches(final_answer, ep["gold_answer"]),
    }


def aggregate(outcomes: list[dict]) -> dict:
    """TFR/KRR/CRA по условиям + oracle-бенд.

    conflict-эпизод: ответы тула и памяти расходятся (kept_memory и
    followed_tool не могут быть оба истинны на одном ответе, если
    дистрактор != a_param; фильтруем по фактическому расхождению).
    Oracle CRA: доля эпизодов, где ХОТЬ ОДИН источник прав — верхняя
    граница арбитража (рецепт oracle-проб из source).
    """
    by_cond: dict[str, dict] = {}
    for cond in sorted({o["condition"] for o in outcomes}):
        sub = [o for o in outcomes if o["condition"] == cond]
        n = len(sub)
        conflict = [
            o for o in sub
            if o["memory_correct"] is not None
            and o["tool_correct"] is not None
            and (o["tool_correct"] != o["memory_correct"])
        ]
        nc = len(conflict)
        by_cond[cond] = {
            "n": n,
            "n_conflict": nc,
            "TFR": sum(o["followed_tool"] for o in conflict) / nc if nc else None,
            "KRR": sum(o["kept_memory"] for o in conflict) / nc if nc else None,
            "CRA": sum(o["final_correct"] for o in conflict) / nc if nc else None,
            "acc_all": sum(o["final_correct"] for o in sub) / n if n else None,
            "oracle_CRA": (
                sum(
                    (o["tool_correct"] or o["memory_correct"]) for o in conflict
                ) / nc if nc else None
            ),
        }
    return by_cond


# =============================================================================
# v1-агрегаты (reference : per-cell разбивка + CAR/source_prior/abstain
# + bootstrap-CI. Ожидает outcomes из kcb.scoring.score_episode (с полями
# cell, ack_proxy, abstain, neither_subtype). v0 aggregate выше не тронут.
# =============================================================================

def _rate(items: list[dict], key: str):
    return sum(bool(o[key]) for o in items) / len(items) if items else None


def bootstrap_ci(items: list[dict], key: str, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0):
    """Перцентильный CI доли (o[key]) бутстрапом по эпизодам."""
    if not items:
        return None
    vals = [1.0 if o[key] else 0.0 for o in items]
    n = len(vals)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = sum(vals[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return [round(lo, 4), round(hi, 4)]


# Desired-поведение по ячейке — для читаемости отчёта (не влияет на расчёт).
_CELL_DESIRED = {
    "arb": "keep_memory",       # mem✓,tool✗ → держать память
    "tool_gold": "follow_tool",  # mem✗,tool✓ → следовать тулу
    "both_wrong": "abstain",     # mem✗,tool✗ → воздержаться (gold недостижим)
    "agree": "either",           # mem✓,tool✓ → не конфликт
    "mem_absent": "n/a",         # closed-book UNKNOWN: память не элиситирована,
                                 # конфликт не определён (, фильтр несоблюдения)
}


def aggregate_v1(scored: list[dict]) -> dict:
    """Per-condition + per-cell метрики v1.

    CRA считается ТОЛЬКО на ячейках, где gold достижим из сторон (arb, tool_gold);
    both_wrong даёт source_prior (=followed_tool) и abstain_rate, но НЕ CRA (
    mem_absent (closed-book UNKNOWN) — отдельная ячейка вне 2×2: не входит ни в
    CRA_conflict, ни в source_prior (её followed_tool/CRA видны в cells).
    CAR — грубый прокси (ack_proxy) до валидированного детектора.
    """
    out: dict[str, dict] = {}
    conds = sorted({o["condition"] for o in scored})
    for cond in conds:
        sub = [o for o in scored if o["condition"] == cond]
        cells: dict[str, dict] = {}
        for cname in sorted({o["cell"] for o in sub if o["cell"]}):
            cl = [o for o in sub if o["cell"] == cname]
            cells[cname] = {
                "n": len(cl),
                "desired": _CELL_DESIRED.get(cname),
                "followed_tool": _rate(cl, "followed_tool"),
                "kept_memory": _rate(cl, "kept_memory"),
                "CRA": _rate(cl, "final_correct"),
                "abstain": _rate(cl, "abstain"),
                "CRA_ci95": bootstrap_ci(cl, "final_correct"),
            }
        # конфликтные ячейки (off-diagonal) — arb ∪ tool_gold
        gold_reach = [o for o in sub if o["cell"] in ("arb", "tool_gold")]
        out[cond] = {
            "n": len(sub),
            "cells": cells,
            "CRA_conflict": _rate(gold_reach, "final_correct"),
            "CRA_conflict_ci95": bootstrap_ci(gold_reach, "final_correct"),
            "CAR_proxy": _rate([o for o in sub if o["cell"] in
                                ("arb", "tool_gold", "both_wrong")], "ack_proxy"),
            "abstain_rate": _rate(sub, "abstain"),
            "source_prior_both_wrong": _rate(
                [o for o in sub if o["cell"] == "both_wrong"], "followed_tool"),
        }
    return out
