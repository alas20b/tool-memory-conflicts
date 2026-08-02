"""Extraction-first скоринг v1 (reference

Дефект v0 (run): наивный матчинг конфаундил СТИЛЬ — многословные ответы
проваливали token-F1 против короткой сущности и инвертировали межмодельные
сравнения. Фикс: сначала ИЗВЛЕЧЬ спан (модель заканчивает строкой
`FINAL: <answer>`), потом матчить алиасо-толерантным bidir-матчером (основной
для v1; v0-substring/f1 остаются как sensitivity в matcher_sensitivity.py).

Здесь же — таксономия исхода {followed_tool, kept_memory, both, neither} с
подтипом neither (abstain/other) и разметка ячейки 2×2
{agree, arb, tool_gold, both_wrong} по правоте памяти×тула (,
Параметрическая сторона тоже extraction-first (: a_param = извлечённый спан);
closed-book UNKNOWN/отказ — фильтр несоблюдения → отдельная ячейка `mem_absent`
(не смешивается с both_wrong: «памяти нет» ≠ «память неправа»).
abstain меряется по FINAL-спану, не по сырому ответу: TOOL_ERROR_PAYLOAD
содержит 'unavailable', и эхо пейлоада в рассуждении давало ложный abstain
(run: raw 0.64 vs span 0.22); сырой вариант остаётся как abstain_raw.
CAR пока — ГРУБЫЙ прокси (both = совпал и с тулом, и с памятью); валидированный
детектор (judge+regex, сверка на ≥100) — отдельный шаг (

v0 kcb/metrics.py НЕ трогается (репро run–run).
"""

from __future__ import annotations

import re

from .metrics import normalize  # общий нормализатор (NFKD, lower, пунктуация)

# --- извлечение ответа --------------------------------------------------------

_FINAL_RE = re.compile(r"final\s*:\s*", re.IGNORECASE)


# Эхо плейсхолдера из шаблона «FINAL: <answer>» (run: gemma копирует его
# буквально в 325/535 ответов, реальный ответ — строкой ВЫШЕ FINAL).
_PLACEHOLDER_RE = re.compile(r"^<answer>\s*(.*)$", re.IGNORECASE)


def extract_final(text: str) -> str:
    """Спан после последнего 'FINAL:'; фолбэк — последняя непустая строка.

    Если спан — буквальный плейсхолдер '<answer>' (эхо шаблона), берём текст
    после него, а если его нет — последнюю непустую строку ДО маркера FINAL.
    """
    if not text:
        return ""
    hits = list(_FINAL_RE.finditer(text))
    if hits:
        tail = text[hits[-1].end():]
        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        span = lines[0] if lines else ""
        ph = _PLACEHOLDER_RE.match(span)
        if ph is not None:
            rest = ph.group(1).strip()
            if rest:
                return rest
            head_lines = [ln.strip() for ln in text[:hits[-1].start()].splitlines()
                          if ln.strip()]
            return head_lines[-1] if head_lines else ""
        return span
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


# --- матчер (bidir + лёгкие алиасы) -------------------------------------------

_ARTICLES = {"the", "a", "an"}
_ALIAS = {"st": "saint", "mt": "mount"}  # '&' бесполезен: normalize режет пунктуацию


def _canon(s: str) -> str:
    toks = [_ALIAS.get(t, t) for t in normalize(s).split() if t not in _ARTICLES]
    return " ".join(toks)


def match(answer: str, target: str) -> bool:
    """Алиасо-толерантное двустороннее вхождение (основной матчер v1)."""
    a, t = _canon(answer), _canon(target)
    return bool(a) and bool(t) and (t in a or a in t)


# --- воздержание --------------------------------------------------------------

_ABSTAIN_RE = re.compile(
    r"\b(unknown|do(?:es)?\s*n'?o?t\s*know|"
    r"can(?:not|'?t)\s*(?:answer|determine|provide|say|tell)|"
    r"no\s*(?:information|answer|idea)|not\s*(?:available|sure)|unavailable|"
    r"unable\s*to|n/?a)\b"
    # v3 (run/B2): пассивные и безличные формы отказа. Валидировано об
    # stage_final (0 срабатываний на human kept/followed из 120) + recall на
    # gemma neither-other (45/57 пойманы, остаток — третьи сущности).
    # Хеджи "more information is needed" ПОСЛЕ ответа не должны стрелять →
    # need-more-формы заякорены на начало спана (\A).
    r"|(?:\A\s*(?:i\s+)?need\s+more\s+information"
    r"|\A\s*more\s+information\s+is\s+(?:needed|required)"
    r"|\b(?:(?:cannot|can\s*not|could\s*not)\s+be\s+"
    r"(?:determined|retrieved|found|verified|identified|established)"
    r"|does\s+not\s+(?:provide|specify|identify|mention|contain|allow|state|answer)"
    r"|not\s+(?:provided|specified|mentioned|found|determined"
    r"|possible\s+to\s+determine)"
    r"|insufficient\s+(?:information|data)"
    r"|(?:information|data)\s+is\s+insufficient"
    r"|insufficient\s+to\s+(?:answer|determine)"
    r"|do(?:es)?\s+not\s+have\s+enough\s+information"
    r"|is\s+not\s+relevant\s+to\s+the\s+question"
    r"|no\s+exact\s+match)\b)",
    re.IGNORECASE,
)


def is_abstain(text: str) -> bool:
    return bool(_ABSTAIN_RE.search(text or ""))


# --- исход и ячейка -----------------------------------------------------------

def classify_outcome(raw_final: str, tool_value, parametric_span: str, gold: str) -> dict:
    """Разбор одного ответа: за кем пошла модель, права ли, вскрыла ли конфликт.

    parametric_span — уже ИЗВЛЕЧЁННЫЙ closed-book ответ (не сырой текст):
    матчинг по сырому тексту ловил gold из рассуждения при другом FINAL.
    """
    fin = extract_final(raw_final)
    ft = tool_value is not None and match(fin, str(tool_value))
    km = bool(parametric_span) and match(fin, parametric_span)
    fc = match(fin, gold)
    if ft and km:
        outcome = "both"
    elif ft:
        outcome = "followed_tool"
    elif km:
        outcome = "kept_memory"
    else:
        outcome = "neither"
    abstain = is_abstain(fin)  # по спану: эхо error-пейлоада в теле ≠ отказ
    subtype = None
    if outcome == "neither":
        subtype = "abstain" if abstain else "other"
    return {
        "extracted": fin,
        "followed_tool": ft,
        "kept_memory": km,
        "final_correct": fc,
        "outcome": outcome,
        "neither_subtype": subtype,
        "ack_proxy": ft and km,   # ГРУБЫЙ прокси CAR — заменить детектором
        "abstain": abstain,
        "abstain_raw": is_abstain(raw_final),  # старое поведение, sensitivity
    }


_CELL = {
    (True, True): "agree",
    (True, False): "arb",
    (False, True): "tool_gold",
    (False, False): "both_wrong",
}


def cell(memory_correct, tool_correct):
    """Ячейка квадранта 2×2 по правоте памяти×тула.

    None при no_tool (tool_correct=None) и при неэлиситированной памяти
    (memory_correct=None — вызывающий помечает mem_absent). tool_error несёт
    tool_correct=False → ячейки размечаются (mem✓ → arb: desired fallback).
    """
    if memory_correct is None or tool_correct is None:
        return None
    return _CELL[(bool(memory_correct), bool(tool_correct))]


def score_episode(ep: dict, final_answer: str, parametric_answer: str) -> dict:
    """Полная разметка эпизода v1: outcome + ячейка + правоты.

    Параметрика: a_param = extract_final(closed-book) (; UNKNOWN/отказ/пусто →
    mem_absent=True, memory_correct=None, ячейка 'mem_absent' (фильтр
    несоблюдения  — эти эпизоды не входят в 2×2 и в source_prior).
    """
    tool_out = ep.get("tool_output") or {}
    tool_value = tool_out.get("result") if isinstance(tool_out, dict) else None
    p_span = extract_final(parametric_answer) if parametric_answer else ""
    mem_absent = (not p_span) or is_abstain(p_span)
    mem_correct = None if mem_absent else match(p_span, ep["gold_answer"])
    oc = classify_outcome(final_answer, tool_value,
                          "" if mem_absent else p_span, ep["gold_answer"])
    c = cell(mem_correct, ep["tool_correct"])
    if mem_absent and ep["tool_correct"] is not None:
        c = "mem_absent"
    return {
        "episode_id": ep["episode_id"],
        "condition": ep["condition"],
        "tau": ep.get("tau"),
        "divergence_bucket": ep.get("divergence_bucket"),
        "tool_correct": ep["tool_correct"],
        "memory_correct": mem_correct,
        "mem_absent": mem_absent,
        "parametric_span": p_span,
        "cell": c,
        **oc,
    }
