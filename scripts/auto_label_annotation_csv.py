"""Auto-label semantic columns in pre-filled annotation CSV files.

This is a conservative assistant for AliMeeting-derived annotation sheets.  It
does not try to recover real names from anonymous speaker IDs.  Instead it fills
the semantic columns with rule-based labels that annotators can review:

* high-overlap rows are always ``uncertainty``;
* low-overlap rows are classified as decision/action/open_question/etc.;
* only the formal semantic columns from issue #65 are written:
  ``event_type``, ``content``, ``owner``, ``deadline``, and
  ``uncertainty_note``;
* action-item owners are speaker IDs only when the text is a first-person
  commitment by the current speaker or explicitly mentions the exact current
  speaker ID, otherwise ``uncertain``;
* deadline extraction mirrors the lightweight rule baseline patterns.

Usage::

    python scripts/auto_label_annotation_csv.py data/annotations/prefilled/*.csv
    python scripts/auto_label_annotation_csv.py --overwrite data/annotations/prefilled/*.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

DECISION_CUES = (
    "决定", "决议", "敲定", "拍板", "通过", "批准", "同意采用", "达成一致",
    "decided", "decision", "we will go with", "let's go with", "lets go with",
    "agreed to", "we agree", "approve", "approved", "go ahead with", "final call",
)
ACTION_CUES = (
    "负责", "跟进", "待办", "需要完成", "请务必", "务必", "提交", "安排", "落实",
    "action item", "action:", "todo", "to-do", "to do", "follow up", "follow-up",
    "will handle", "take care of", "assigned to", "assign", "deliver", "prepare",
    "please", "need to", "responsible for",
)
OPEN_QUESTION_CUES = (
    "待确认", "待定", "需要确认", "还不清楚", "尚未确定", "是否", "有待讨论",
    "open question", "tbd", "to be decided", "to be determined", "not sure whether",
    "unclear whether", "question remains", "still open",
)

DEADLINE_PATTERNS = (
    re.compile(r"\d{4}-\d{1,2}-\d{1,2}"),
    re.compile(r"\d{1,2}\s*[月/]\s*\d{1,2}\s*[日号]?"),
    re.compile(r"(今天|明天|后天|大后天|本周[一二三四五六日天]?|下周[一二三四五六日天]?|"
               r"周[一二三四五六日天]|月底|本月底|下个?月底?)"),
    re.compile(r"(?i)\b(today|tomorrow|tonight|this week|next week|next month|"
               r"end of (?:day|week|month)|eod|eow)\b"),
    re.compile(r"(?i)\bby\s+(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|"
               r"saturday|sunday|\w+day|the\s+\d{1,2}(?:st|nd|rd|th)?)\b"),
)

LEGACY_SEMANTIC_FIELDS = (
    "topic",
    "decision",
    "action_item",
)

LABEL_FIELDS = (
    "event_type",
    "content",
    "owner",
    "deadline",
    "uncertainty_note",
)

SEMANTIC_FIELDS = LEGACY_SEMANTIC_FIELDS + LABEL_FIELDS

UNCERTAINTY_CONTENT = "高重叠语音，内容或说话人归属不确定"
UNCERTAINTY_NOTE = "多人同时说话，按项目规则标为不确定"
PURE_ACKNOWLEDGEMENTS = {"对", "同意", "可以", "好", "行"}

FIRST_PERSON_ACTION_PATTERNS = (
    re.compile(r"(我|我们|咱们)(来|会|负责|跟进|处理|完成|整理|提交|确认|安排|落实|做|测试|修改|补充|对接|推进)"),
    re.compile(r"(由|让)?(我|我们|咱们)(这边)?(来|负责|跟进|处理|完成|整理|提交|确认|安排|落实|做)"),
    re.compile(r"(?i)\b(i|we)\s*(will|can|shall|'ll)\b"),
    re.compile(r"(?i)\b(i|we)\s*(am|are)\s+(responsible|going to)\b"),
)

ASSIGNED_TO_OTHER_PATTERNS = (
    re.compile(r"(你|你们|他|她|他们|她们|大家|产品|测试|研发|开发|设计|运营|市场|销售|法务|财务|同学|团队)(来|负责|跟进|处理|完成|整理|提交|确认|安排|落实|做)"),
    re.compile(r"[\u4e00-\u9fff]{2,4}(来|负责|跟进|处理|完成|整理|提交|确认|安排|落实|做)"),
    re.compile(r"(?i)\b(you|he|she|they|alice|bob|team|product|qa|dev|design)\s+(will|should|can|need to|needs to)\b"),
)

FILLER_RE = re.compile(
    r"(嗯|呃|啊|这个|那个|就是说|然后|其实|基本上|可能就是说|我觉得|我感觉|"
    r"咱们|我们|大家|一下|这一块儿|这一块|的话)"
)


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "t", "是"}


def _contains_cue(text: str, cues: Iterable[str]) -> bool:
    lowered = text.lower()
    for cue in cues:
        if cue.isascii():
            if cue in lowered:
                return True
        elif cue in text:
            return True
    return False


def classify_event_type(text: str) -> str:
    if _contains_cue(text, DECISION_CUES):
        return "decision"
    if _contains_cue(text, ACTION_CUES):
        return "action_item"
    if text.rstrip().endswith(("?", "？")) or _contains_cue(text, OPEN_QUESTION_CUES):
        return "open_question"
    return "speaker_stance"


def extract_deadline(text: str) -> str | None:
    for pattern in DEADLINE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0).strip()
    return None


def _compact_text(text: str) -> str:
    compact = " ".join(str(text).split())
    compact = FILLER_RE.sub("", compact)
    compact = re.sub(r"[，,。；;：:\s]+", "，", compact).strip("，。；;：: ")
    return compact or " ".join(str(text).split()).strip()


def _short_content(text: str, event_type: str) -> str:
    compact = _compact_text(text)
    if event_type == "decision":
        compact = re.sub(r"^(经过讨论)?(最终)?(决定|决议|确定|确认|敲定|通过|批准)", "", compact)
        prefix = "确定"
    elif event_type == "action_item":
        compact = re.sub(r"^(请|麻烦|需要|务必|安排|落实)", "", compact)
        prefix = "跟进"
    elif event_type == "open_question":
        prefix = "待确认"
    elif event_type == "topic_transition":
        prefix = "讨论"
    elif event_type == "disagreement":
        prefix = "存在分歧"
    else:
        prefix = "认为"
    compact = compact.strip("，。；;：: ")
    if not compact:
        compact = "该片段内容"
    if len(compact) > 44:
        compact = compact[:44].rstrip("，。；;：: ") + "..."
    return compact if compact.startswith(prefix) else f"{prefix}{compact}"


def _is_topic_transition(text: str, event_type: str) -> bool:
    if event_type != "speaker_stance":
        return False
    cues = (
        "今天", "首先", "接下来", "下面", "然后咱们", "我们讨论", "咱们讨论",
        "开始", "进入", "看一下", "说一下", "谈一下", "总结一下",
    )
    return any(cue in text for cue in cues)


def _is_disagreement(text: str, event_type: str) -> bool:
    if event_type != "speaker_stance":
        return False
    return any(cue in text for cue in ("不同意", "不太同意", "不是", "但是我觉得", "反对", "有问题"))


def _mentions_exact_speaker_id(text: str, speaker: str) -> bool:
    if not speaker:
        return False
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(speaker)}(?![A-Za-z0-9_])")
    return bool(pattern.search(text))


def _owner_for_action(text: str, speaker: str) -> str:
    speaker = speaker.strip()
    if _mentions_exact_speaker_id(text, speaker):
        return speaker
    if any(pattern.search(text) for pattern in FIRST_PERSON_ACTION_PATTERNS):
        return speaker or "uncertain"
    if any(pattern.search(text) for pattern in ASSIGNED_TO_OTHER_PATTERNS):
        return "uncertain"
    return "uncertain"


def _is_pure_acknowledgement(text: str) -> bool:
    return text.strip().lower().strip("。！？!?，, ") in PURE_ACKNOWLEDGEMENTS


def label_row(row: dict[str, str]) -> dict[str, str]:
    labeled = dict(row)
    text = row.get("text", "").strip()
    is_overlap = _truthy(row.get("is_overlap", "")) or row.get("overlap_type", "").strip().lower() in {
        "partial",
        "full",
    }

    for field in SEMANTIC_FIELDS:
        labeled[field] = ""

    if is_overlap:
        labeled["event_type"] = "uncertainty"
        labeled["content"] = UNCERTAINTY_CONTENT
        labeled["uncertainty_note"] = UNCERTAINTY_NOTE
        return labeled

    event_type = classify_event_type(text)
    if _is_topic_transition(text, event_type):
        event_type = "topic_transition"
    elif _is_disagreement(text, event_type):
        event_type = "disagreement"

    # Keep pure acknowledgements as speaker stances, not decisions.
    if event_type == "decision" and _is_pure_acknowledgement(text):
        event_type = "speaker_stance"

    content = _short_content(text, event_type)
    labeled["event_type"] = event_type
    labeled["content"] = content

    if event_type == "action_item":
        labeled["owner"] = _owner_for_action(text, row.get("speaker", ""))
        deadline = extract_deadline(text)
        if deadline:
            labeled["deadline"] = deadline
    elif event_type == "open_question" and _contains_cue(text, OPEN_QUESTION_CUES):
        labeled["uncertainty_note"] = "该问题仍待确认"

    return labeled


def label_csv(path: Path, out_path: Path) -> tuple[int, dict[str, int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        fieldnames = list(reader.fieldnames)
        missing = [field for field in SEMANTIC_FIELDS if field not in fieldnames]
        if missing:
            raise ValueError(f"{path}: missing expected columns {missing}")
        rows = [label_row({k: v or "" for k, v in row.items()}) for row in reader]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        event_type = row.get("event_type", "")
        counts[event_type] = counts.get(event_type, 0) + 1
    return len(rows), counts


def output_path_for(path: Path, overwrite: bool) -> Path:
    if overwrite:
        return path
    if path.stem.endswith("_ai_labeled"):
        return path
    return path.with_name(f"{path.stem}_ai_labeled{path.suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", type=Path, help="pre-filled CSV file(s)")
    parser.add_argument("--overwrite", action="store_true", help="write labels back to the input file(s)")
    parser.add_argument("--force", action="store_true", help="overwrite existing *_ai_labeled.csv outputs")
    args = parser.parse_args(argv)

    for path in args.csv:
        if not path.exists():
            raise FileNotFoundError(path)
        out_path = output_path_for(path, args.overwrite)
        if out_path.exists() and out_path != path and not args.force:
            print(f"[skip] {out_path} exists (pass --force to replace)")
            continue
        total, counts = label_csv(path, out_path)
        summary = ", ".join(f"{key or '<blank>'}={value}" for key, value in sorted(counts.items()))
        print(f"[ok] {path.name}: {total} rows -> {out_path} ({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
