from __future__ import annotations

import re
from typing import Any, Dict, List

from ..constants import GRADING_SPEC_VERSION, MC_MODE_STRICT, MC_MODE_WITH_RATIONALE
from ..data import is_multiple_choice_base, multiple_choice_option_map


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_ANSWER_PREFIX_RE = re.compile(
    r"^(?:(?:the\s+)?(?:final\s+)?answer|(?:the\s+)?correct\s+answer)\s*:\s*",
    flags=re.IGNORECASE,
)
_LEADING_FILLER_RE = re.compile(
    r"^(?:(?:the\s+)?answer\s+is|it\s+is|it's|its|this\s+is|that\s+is)\s+",
    flags=re.IGNORECASE,
)
_MULTI_CANDIDATE_SPLIT_RE = re.compile(r"\s+(?:or|/)\s+", flags=re.IGNORECASE)
_APPOSITIVE_SPLIT_RE = re.compile(r",\s+(?:the|a|an)\s+", flags=re.IGNORECASE)
_MC_SINGLE_LABEL_CLASS = r"[A-Za-z0-9]"
_MC_ANSWER_PREFIX_RE = re.compile(
    r"^(?:(?:the\s+)?(?:final\s+)?answer|(?:the\s+)?correct\s+answer|"
    r"(?:the\s+)?correct\s+(?:option|choice)|option|choice)\s*(?:is|:)?\s*",
    flags=re.IGNORECASE,
)
_MC_SEGMENT_PREFIX_RE = re.compile(
    r"^(?:(?:therefore|thus|so|hence|overall|ultimately|finally|probably|maybe|perhaps)[,:]?\s+|"
    r"i\s+(?:pick|choose|select|guess|think)\s+|"
    r"my\s+(?:pick|choice|answer)\s+(?:is\s+)?)",
    flags=re.IGNORECASE,
)
_MC_INLINE_ANSWER_RE = re.compile(
    r"\b(?:(?:the\s+)?(?:final\s+)?answer|(?:the\s+)?correct\s+answer|"
    r"(?:the\s+)?correct\s+(?:option|choice)|answer)\s*(?:is|:)\s*(.+)$",
    flags=re.IGNORECASE,
)
_MC_INLINE_CHOICE_RE = re.compile(
    rf"\b(?:choice|option)\s*\(?({_MC_SINGLE_LABEL_CLASS})\)?\b",
    flags=re.IGNORECASE,
)
_MC_STANDALONE_LETTER_SEGMENT_RE = re.compile(
    rf"^\(?({_MC_SINGLE_LABEL_CLASS})\)?[\]\).,:;-]?$"
)
_STRICT_MC_EXPLICIT_PREFIX_RE = re.compile(
    r"^(?:(?:the\s+)?(?:final\s+)?answer|(?:the\s+)?correct\s+answer|"
    r"(?:the\s+)?correct\s+(?:option|choice)|answer|option|choice)\s*(?:is|:)?\s*(.+)$",
    flags=re.IGNORECASE,
)
_STRICT_MC_ANSWER_LINE_RE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", flags=re.IGNORECASE)
_STRICT_MC_EXACT_LINE_RE = re.compile(
    rf"^\s*answer\s*:\s*({_MC_SINGLE_LABEL_CLASS})\s*$",
    flags=re.IGNORECASE,
)
_STRICT_MC_NONCANONICAL_EXPLICIT_RE = re.compile(
    r"\b(?:the\s+answer\s+is|so\s+the\s+answer\s+is|final\s+answer\s*[:]?|correct\s+answer\s*[:]?|"
    rf"option\s+\(?{_MC_SINGLE_LABEL_CLASS}\)?|choice\s+\(?{_MC_SINGLE_LABEL_CLASS}\)?)\b",
    flags=re.IGNORECASE,
)
_ANSWER_MARKER_RE = re.compile(r"\banswer\s*:", flags=re.IGNORECASE)


def _dedupe_nonempty_strings(values: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _extract_truthful_qa_gold_answers(base: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []

    correct_answer = base.get("correct_answer")
    if isinstance(correct_answer, str):
        aliases.append(correct_answer)

    long_correct_answer = base.get("long_correct_answer")
    if isinstance(long_correct_answer, str):
        aliases.append(long_correct_answer)

    letters = base.get("letters")
    correct_letter = base.get("correct_letter")
    answers_list = base.get("answers_list")
    if (
        isinstance(letters, str)
        and isinstance(correct_letter, str)
        and isinstance(answers_list, list)
        and correct_letter in letters
    ):
        idx = letters.index(correct_letter)
        if 0 <= idx < len(answers_list) and isinstance(answers_list[idx], str):
            aliases.append(answers_list[idx])

    return _dedupe_nonempty_strings(aliases)


def _extract_multiple_choice_gold_answers(base: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []

    correct_answer = base.get("correct_answer")
    if isinstance(correct_answer, str):
        aliases.append(correct_answer)

    long_correct_answer = base.get("long_correct_answer")
    if isinstance(long_correct_answer, str):
        aliases.append(long_correct_answer)

    correct_letter = str(base.get("correct_letter", "") or "").strip()
    option_map = multiple_choice_option_map(base)
    if correct_letter and correct_letter in option_map:
        aliases.append(option_map[correct_letter])

    return _dedupe_nonempty_strings(aliases)


def extract_gold_answers_from_base(base: Dict[str, Any]) -> List[str]:
    if not isinstance(base, dict):
        return []

    if is_multiple_choice_base(base):
        aliases = _extract_multiple_choice_gold_answers(base)
        if aliases:
            return aliases

    dataset_name = str(base.get("dataset", "")).strip().lower()
    if dataset_name == "truthful_qa" or (
        "answers_list" in base and "correct_letter" in base and "letters" in base
    ):
        aliases = _extract_truthful_qa_gold_answers(base)
        if aliases:
            return aliases

    if dataset_name == "trivia_qa" and isinstance(base.get("answer"), list):
        aliases = [item for item in base["answer"] if isinstance(item, str)]
        correct_answer = base.get("correct_answer")
        if isinstance(correct_answer, str):
            aliases.append(correct_answer)
        aliases = _dedupe_nonempty_strings(aliases)
        if aliases:
            return aliases

    for key in ["answer", "gold", "label", "target", "gold_answer", "ground_truth", "answers"]:
        if key not in base:
            continue
        value = base[key]
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            aliases: List[str] = []
            if "value" in value and isinstance(value["value"], str):
                aliases.append(value["value"])
            if "aliases" in value and isinstance(value["aliases"], list):
                aliases.extend([alias for alias in value["aliases"] if isinstance(alias, str)])
            if "normalized_aliases" in value and isinstance(value["normalized_aliases"], list):
                aliases.extend(
                    [alias for alias in value["normalized_aliases"] if isinstance(alias, str)]
                )
            aliases = _dedupe_nonempty_strings(aliases)
            if aliases:
                return aliases
        if isinstance(value, list):
            aliases = []
            for item in value:
                if isinstance(item, str):
                    aliases.append(item)
                elif isinstance(item, dict):
                    if "value" in item and isinstance(item["value"], str):
                        aliases.append(item["value"])
                    if "aliases" in item and isinstance(item["aliases"], list):
                        aliases.extend([alias for alias in item["aliases"] if isinstance(alias, str)])
            aliases = _dedupe_nonempty_strings(aliases)
            if aliases:
                return aliases

    if "answer_aliases" in base and isinstance(base["answer_aliases"], list):
        aliases = [alias for alias in base["answer_aliases"] if isinstance(alias, str)]
        aliases = _dedupe_nonempty_strings(aliases)
        if aliases:
            return aliases

    return []


def normalize_answer(text: str) -> str:
    normalized = text.strip().lower()
    normalized = _WS_RE.sub(" ", normalized)
    normalized = _PUNCT_RE.sub("", normalized)
    normalized = _WS_RE.sub(" ", normalized).strip()
    return normalized


def _candidate_matches_gold(pred: str, gold_answers: List[str]) -> bool:
    normalized_pred = normalize_answer(pred)
    if not normalized_pred:
        return False
    gold_norm = [normalize_answer(gold) for gold in gold_answers if gold]

    if normalized_pred in set(gold_norm):
        return True

    for gold in gold_norm:
        if not gold:
            continue
        if re.search(rf"(?:^|\s){re.escape(gold)}(?:$|\s)", normalized_pred):
            return True

    return False


def _candidate_matches_option_text(pred: str, option_text: str) -> bool:
    normalized_pred = normalize_answer(pred)
    normalized_option = normalize_answer(option_text)
    if not normalized_pred or not normalized_option:
        return False
    if normalized_pred == normalized_option:
        return True
    return re.search(rf"(?:^|\s){re.escape(normalized_pred)}(?:$|\s)", normalized_option) is not None


def _grading_result(
    *,
    parsed_answer: str,
    correctness: Any,
    status: str,
    reason: str,
    usable_for_metrics: bool,
    committed_answer: str = "",
    commitment_kind: str = "none",
    commitment_source: str = "",
    starts_with_answer_prefix: bool = False,
    strict_format_exact: bool = False,
    commitment_line: str = "",
    answer_marker_count: int = 0,
    multiple_answer_markers: bool = False,
) -> Dict[str, Any]:
    return {
        "parsed_answer": parsed_answer,
        "correctness": correctness,
        "status": status,
        "reason": reason,
        "usable_for_metrics": usable_for_metrics,
        "committed_answer": committed_answer,
        "commitment_kind": commitment_kind,
        "commitment_source": commitment_source,
        "starts_with_answer_prefix": bool(starts_with_answer_prefix),
        "strict_format_exact": bool(strict_format_exact),
        "commitment_line": commitment_line,
        "answer_marker_count": int(answer_marker_count),
        "multiple_answer_markers": bool(multiple_answer_markers),
        "grading_spec_version": int(GRADING_SPEC_VERSION),
    }


def _multiple_choice_mode(base: Dict[str, Any]) -> str:
    if not isinstance(base, dict):
        return ""
    mc_mode = str(base.get("mc_mode", "") or "").strip().lower()
    if mc_mode in {MC_MODE_STRICT, MC_MODE_WITH_RATIONALE}:
        return mc_mode
    answer_channel = str(base.get("answer_channel", "") or "").strip().lower()
    if answer_channel == "letter":
        return MC_MODE_STRICT
    return ""


def extract_short_answer_from_generation(text: str) -> str:
    short = text.strip()
    if not short:
        return ""

    match = _ANSWER_PREFIX_RE.search(short)
    if match:
        short = short[match.end() :].strip()

    lines = short.splitlines()
    if not lines:
        return ""
    short = lines[0].strip()
    short = re.split(r"[.?!]\s+", short, maxsplit=1)[0].strip()
    short = _APPOSITIVE_SPLIT_RE.split(short, maxsplit=1)[0].strip()
    short = _LEADING_FILLER_RE.sub("", short).strip()
    short = short.strip(" \"'“”‘’\t")

    return short


def _extract_answer_candidates(text: str) -> List[str]:
    if not text:
        return []

    parts = [part.strip(" \"'“”‘’\t") for part in _MULTI_CANDIDATE_SPLIT_RE.split(text)]
    parts = [part for part in parts if part]
    if not parts:
        return []

    deduped: List[str] = []
    seen = set()
    for part in parts:
        normalized = normalize_answer(part)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(part)
    return deduped


def _extract_multiple_choice_letter(parsed_answer: str, letters: str) -> str:
    letters = str(letters or "").strip().upper()
    if not parsed_answer or not letters:
        return ""

    allowed = set(letters)
    text = parsed_answer.strip()
    text = _MC_ANSWER_PREFIX_RE.sub("", text).strip()
    text = text.strip(" \"'“”‘’\t")

    if _is_ambiguous_multiple_choice_letter_sequence(text, letters):
        return ""

    patterns = [
        rf"^\(?({_MC_SINGLE_LABEL_CLASS})\)?$",
        rf"^\(?({_MC_SINGLE_LABEL_CLASS})\)?[\s\]\).,:;-].*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text)
        if not match:
            continue
        letter = match.group(1).upper()
        if letter in allowed:
            return letter
    return ""


def _iter_reversed_output_segments(text: str) -> List[str]:
    segments: List[str] = []
    seen = set()

    for segment in reversed(text.splitlines()):
        stripped = segment.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        segments.append(stripped)

    for segment in reversed(re.split(r"(?<=[.?!])\s+", text.strip())):
        stripped = segment.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        segments.append(stripped)

    return segments


def _is_ambiguous_multiple_choice_letter_sequence(text: str, letters: str) -> bool:
    allowed = set(str(letters or "").strip().upper())
    candidate = _MC_ANSWER_PREFIX_RE.sub("", str(text or "").strip()).strip(" \"'“”‘’\t")
    candidate = candidate.rstrip(".,;:!-")
    if not candidate:
        return False
    if not re.fullmatch(
        rf"\(?{_MC_SINGLE_LABEL_CLASS}\)?(?:\s*(?:or|/)\s*\(?{_MC_SINGLE_LABEL_CLASS}\)?)+",
        candidate,
        flags=re.IGNORECASE,
    ):
        return False

    parts = [part.strip().strip("()").upper() for part in re.split(r"\s*(?:or|/)\s*", candidate, flags=re.IGNORECASE)]
    return len(parts) > 1 and all(len(part) == 1 and part in allowed for part in parts)


def _strip_multiple_choice_segment_prefixes(text: str) -> str:
    candidate = text.strip(" \"'“”‘’\t")
    while candidate:
        updated = _MC_SEGMENT_PREFIX_RE.sub("", candidate).strip(" \"'“”‘’\t")
        if updated == candidate:
            break
        candidate = updated
    return candidate


def _segment_looks_like_multiple_choice_answer(
    segment: str,
    letters: str,
    gold_answers: List[str],
    option_map: Dict[str, str],
) -> bool:
    if not segment:
        return False

    if _extract_multiple_choice_letter(segment, letters):
        return True

    candidates = _extract_answer_candidates(segment)
    if not candidates:
        return False

    for candidate in candidates:
        if _extract_multiple_choice_letter(candidate, letters):
            continue
        if _candidate_matches_gold(candidate, gold_answers):
            continue
        matching_letters = {
            option_letter
            for option_letter, option_text in option_map.items()
            if _candidate_matches_option_text(candidate, option_text)
        }
        if len(matching_letters) == 1:
            continue
        return False
    return True


def _extract_multiple_choice_candidate_from_full_output(
    text: str,
    letters: str,
    gold_answers: List[str],
    option_map: Dict[str, str],
) -> str:
    if not text:
        return ""

    segments = _iter_reversed_output_segments(text)
    for segment in segments:
        match = _MC_INLINE_ANSWER_RE.search(segment)
        if not match:
            continue
        candidate = extract_short_answer_from_generation(match.group(1))
        if candidate:
            return candidate

    for segment in segments:
        stripped_segment = _strip_multiple_choice_segment_prefixes(segment)
        if stripped_segment and _segment_looks_like_multiple_choice_answer(
            stripped_segment,
            letters,
            gold_answers,
            option_map,
        ):
            return stripped_segment

        if _is_ambiguous_multiple_choice_letter_sequence(stripped_segment or segment, letters):
            return stripped_segment or segment

        match = _MC_INLINE_CHOICE_RE.search(segment)
        if match:
            letter = match.group(1).upper()
            if letter in set(str(letters or "").strip().upper()):
                return letter

        stripped = segment.strip(" \"'“”‘’\t")
        match = _MC_STANDALONE_LETTER_SEGMENT_RE.match(stripped)
        if not match:
            continue
        letter = match.group(1).upper()
        if letter in set(str(letters or "").strip().upper()):
            return letter

    return ""


def grade_short_answer(text: str, gold_answers: List[str]) -> Dict[str, Any]:
    parsed_answer = extract_short_answer_from_generation(text)
    if not gold_answers:
        return _grading_result(
            parsed_answer=parsed_answer,
            correctness=None,
            status="ambiguous",
            reason="missing_gold_answers",
            usable_for_metrics=False,
        )
    if not parsed_answer:
        return _grading_result(
            parsed_answer=parsed_answer,
            correctness=None,
            status="ambiguous",
            reason="empty_answer",
            usable_for_metrics=False,
        )

    candidates = _extract_answer_candidates(parsed_answer)
    if not candidates:
        return _grading_result(
            parsed_answer=parsed_answer,
            correctness=None,
            status="ambiguous",
            reason="no_candidate_extracted",
            usable_for_metrics=False,
        )
    if len(candidates) > 1:
        return _grading_result(
            parsed_answer=parsed_answer,
            correctness=None,
            status="ambiguous",
            reason="multiple_candidates",
            usable_for_metrics=False,
        )

    is_correct = _candidate_matches_gold(candidates[0], gold_answers)
    return _grading_result(
        parsed_answer=parsed_answer,
        correctness=int(is_correct),
        status="correct" if is_correct else "incorrect",
        reason="single_candidate_match" if is_correct else "single_candidate_non_match",
        usable_for_metrics=True,
        committed_answer=candidates[0],
        commitment_kind="text",
        commitment_source="first_line_fallback",
    )


def _extract_strict_mc_letter_candidates(text: str, letters: str) -> List[str]:
    allowed = set(str(letters or "").strip().upper())
    candidate = str(text or "").strip().strip(" \"'“”‘’\t")
    candidate = candidate.rstrip(".,;:!?")
    if not candidate:
        return []
    if not re.fullmatch(
        rf"\(?{_MC_SINGLE_LABEL_CLASS}\)?(?:\s*(?:or|/|,)\s*\(?{_MC_SINGLE_LABEL_CLASS}\)?)*",
        candidate,
        flags=re.IGNORECASE,
    ):
        return []
    parsed = [
        part.strip().strip("()").upper()
        for part in re.split(r"\s*(?:or|/|,)\s*", candidate, flags=re.IGNORECASE)
        if part.strip()
    ]
    if not parsed or any(len(part) != 1 or part not in allowed for part in parsed):
        return []
    return list(dict.fromkeys(parsed))


def _strict_mc_nonempty_lines(text: str) -> List[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def _answer_marker_count(text: str) -> int:
    return len(_ANSWER_MARKER_RE.findall(str(text or "")))


def _strict_mc_expected_response_prefix(base: Dict[str, Any]) -> str:
    return str(base.get("response_prefix", "") or "").strip()


def _strict_mc_prefilled_prefix_candidates(text: str, letters: str) -> List[str]:
    lines = _strict_mc_nonempty_lines(text)
    if not lines:
        return []
    return _extract_strict_mc_answer_payload_candidates(lines[0], letters)


def _strict_mc_prefilled_prefix_exact(text: str, letters: str) -> bool:
    lines = _strict_mc_nonempty_lines(text)
    if len(lines) != 1:
        return False
    match = _MC_STANDALONE_LETTER_SEGMENT_RE.match(lines[0])
    if not match:
        return False
    allowed = set(str(letters or "").strip().upper())
    return bool(allowed and match.group(1).upper() in allowed)


def _strict_mc_starts_with_answer_prefix(text: str, letters: str, base: Dict[str, Any]) -> bool:
    lines = _strict_mc_nonempty_lines(text)
    if lines and _STRICT_MC_ANSWER_LINE_RE.match(lines[0]):
        return True
    if _strict_mc_expected_response_prefix(base).lower() == "answer:":
        return bool(_strict_mc_prefilled_prefix_candidates(text, letters))
    return False


def _strict_mc_format_exact(text: str, letters: str, base: Dict[str, Any]) -> bool:
    lines = _strict_mc_nonempty_lines(text)
    if len(lines) == 1 and bool(_STRICT_MC_EXACT_LINE_RE.match(lines[0])):
        return True
    if _strict_mc_expected_response_prefix(base).lower() == "answer:":
        return _strict_mc_prefilled_prefix_exact(text, letters)
    return False


def _extract_strict_mc_answer_payload_candidates(text: str, letters: str) -> List[str]:
    candidate = str(text or "").strip().strip(" \"'“”‘’\t")
    if not candidate:
        return []

    candidates = _extract_strict_mc_letter_candidates(candidate, letters)
    if candidates:
        return candidates

    allowed = set(str(letters or "").strip().upper())
    match = re.match(
        rf"^\(?({_MC_SINGLE_LABEL_CLASS})(?:\)|\])?(?=$|[\s\]\).,:;\-]).*",
        candidate,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    letter = match.group(1).upper()
    if letter not in allowed:
        return []
    return [letter]


def _extract_strict_mc_commitments(
    text: str,
    letters: str,
    base: Dict[str, Any],
) -> List[Dict[str, Any]]:
    commitments: List[Dict[str, Any]] = []
    seen = set()
    lines = _strict_mc_nonempty_lines(text)

    def record_candidates(
        candidates: List[str],
        source: str,
        segment: str,
        *,
        strict_format_exact: bool = False,
    ) -> None:
        if not candidates:
            return
        key = (tuple(candidates), source, segment)
        if key in seen:
            return
        seen.add(key)
        commitments.append(
            {
                "candidates": candidates,
                "source": source,
                "segment": segment.strip(),
                "strict_format_exact": bool(strict_format_exact),
            }
        )

    for idx, line in enumerate(lines):
        explicit_match = _STRICT_MC_ANSWER_LINE_RE.match(line)
        if explicit_match:
            record_candidates(
                _extract_strict_mc_answer_payload_candidates(explicit_match.group(1), letters),
                "explicit_answer_line" if idx == 0 else "late_explicit_answer_line",
                line,
                strict_format_exact=_strict_mc_format_exact(line, letters, base),
            )
    if commitments:
        return commitments

    for idx, line in enumerate(lines):
        record_candidates(
            _extract_strict_mc_letter_candidates(line, letters),
            "standalone_answer_line" if idx == 0 else "late_standalone_answer_line",
            line,
            strict_format_exact=(
                idx == 0
                and _strict_mc_expected_response_prefix(base).lower() == "answer:"
                and _strict_mc_prefilled_prefix_exact(line, letters)
            ),
        )
    return commitments


def _find_noncanonical_explicit_answer(text: str) -> tuple[str, str]:
    lines = _strict_mc_nonempty_lines(text)
    for line in lines:
        if _STRICT_MC_ANSWER_LINE_RE.match(line):
            return line, "explicit_answer_line"
    for line in lines:
        if _STRICT_MC_NONCANONICAL_EXPLICIT_RE.search(line):
            return line, "noncanonical_explicit_line"
    return "", ""


def _grade_multiple_choice_response_strict(
    text: str,
    base: Dict[str, Any],
    generation_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    letters = str(base.get("letters", "") or "").strip()
    correct_letter = str(base.get("correct_letter", "") or "").strip().upper()
    generation_info = generation_info or {}
    parsed_answer = extract_short_answer_from_generation(text)
    starts_with_answer_prefix = _strict_mc_starts_with_answer_prefix(text, letters, base)
    strict_format_exact = _strict_mc_format_exact(text, letters, base)
    answer_marker_count = _answer_marker_count(text)
    multiple_answer_markers = answer_marker_count > 1
    if not correct_letter or not letters:
        return _grading_result(
            parsed_answer=parsed_answer,
            correctness=None,
            status="ambiguous",
            reason="missing_gold_answers",
            usable_for_metrics=False,
            starts_with_answer_prefix=starts_with_answer_prefix,
            strict_format_exact=strict_format_exact,
            answer_marker_count=answer_marker_count,
            multiple_answer_markers=multiple_answer_markers,
        )

    commitments = _extract_strict_mc_commitments(text, letters, base)
    if not commitments:
        noncanonical_line, noncanonical_source = _find_noncanonical_explicit_answer(text)
        no_commit_reason = (
            "noncanonical_explicit_answer"
            if noncanonical_line
            else (
                "truncated_before_commitment"
                if bool(generation_info.get("hit_max_new_tokens", False))
                else "no_committed_answer"
            )
        )
        return _grading_result(
            parsed_answer=noncanonical_line or parsed_answer,
            correctness=None,
            status="ambiguous",
            reason=no_commit_reason,
            usable_for_metrics=False,
            commitment_source=noncanonical_source,
            starts_with_answer_prefix=starts_with_answer_prefix,
            strict_format_exact=strict_format_exact,
            commitment_line=noncanonical_line,
            answer_marker_count=answer_marker_count,
            multiple_answer_markers=multiple_answer_markers,
        )

    first_commitment = commitments[0]
    if len(first_commitment["candidates"]) > 1:
        return _grading_result(
            parsed_answer=first_commitment["segment"],
            correctness=None,
            status="ambiguous",
            reason="multiple_letter_candidates",
            usable_for_metrics=False,
            commitment_kind="ambiguous",
            commitment_source=first_commitment["source"],
            starts_with_answer_prefix=starts_with_answer_prefix,
            strict_format_exact=False,
            commitment_line=first_commitment["segment"],
            answer_marker_count=answer_marker_count,
            multiple_answer_markers=multiple_answer_markers,
        )

    distinct_letters = {
        candidate
        for commitment in commitments
        for candidate in commitment["candidates"]
    }
    if len(distinct_letters) > 1:
        return _grading_result(
            parsed_answer=first_commitment["segment"],
            correctness=None,
            status="ambiguous",
            reason="conflicting_committed_answers",
            usable_for_metrics=False,
            commitment_kind="ambiguous",
            commitment_source=first_commitment["source"],
            starts_with_answer_prefix=starts_with_answer_prefix,
            strict_format_exact=False,
            commitment_line=first_commitment["segment"],
            answer_marker_count=answer_marker_count,
            multiple_answer_markers=multiple_answer_markers,
        )

    committed_answer = first_commitment["candidates"][0]
    is_correct = committed_answer == correct_letter
    return _grading_result(
        parsed_answer=first_commitment["segment"],
        correctness=int(is_correct),
        status="correct" if is_correct else "incorrect",
        reason="single_letter_match" if is_correct else "single_letter_non_match",
        usable_for_metrics=True,
        committed_answer=committed_answer,
        commitment_kind="letter",
        commitment_source=first_commitment["source"],
        starts_with_answer_prefix=starts_with_answer_prefix,
        strict_format_exact=bool(first_commitment.get("strict_format_exact", False)) and strict_format_exact,
        commitment_line=first_commitment["segment"],
        answer_marker_count=answer_marker_count,
        multiple_answer_markers=multiple_answer_markers,
    )


def _grade_multiple_choice_response_legacy(text: str, base: Dict[str, Any]) -> Dict[str, Any]:
    gold_answers = extract_gold_answers_from_base(base)
    letters = str(base.get("letters", "") or "").strip()
    option_map = multiple_choice_option_map(base)
    parsed_answer = _extract_multiple_choice_candidate_from_full_output(
        text,
        letters,
        gold_answers,
        option_map,
    )
    if not parsed_answer:
        parsed_answer = extract_short_answer_from_generation(text)
    if not gold_answers and not str(base.get("correct_letter", "")).strip():
        return _grading_result(
            parsed_answer=parsed_answer,
            correctness=None,
            status="ambiguous",
            reason="missing_gold_answers",
            usable_for_metrics=False,
        )
    if not parsed_answer:
        return _grading_result(
            parsed_answer=parsed_answer,
            correctness=None,
            status="ambiguous",
            reason="empty_answer",
            usable_for_metrics=False,
        )
    if _is_ambiguous_multiple_choice_letter_sequence(parsed_answer, letters):
        return _grading_result(
            parsed_answer=parsed_answer,
            correctness=None,
            status="ambiguous",
            reason="multiple_letter_candidates",
            usable_for_metrics=False,
            commitment_kind="ambiguous",
            commitment_source="full_output_scan",
        )

    correct_letter = str(base.get("correct_letter", "") or "").strip().upper()
    candidates = _extract_answer_candidates(parsed_answer)
    parsed_letter = _extract_multiple_choice_letter(parsed_answer, letters) if len(candidates) <= 1 else ""
    if parsed_letter:
        is_correct = parsed_letter == correct_letter
        return _grading_result(
            parsed_answer=parsed_answer,
            correctness=int(is_correct),
            status="correct" if is_correct else "incorrect",
            reason="single_letter_match" if is_correct else "single_letter_non_match",
            usable_for_metrics=True,
            committed_answer=parsed_letter,
            commitment_kind="letter",
            commitment_source="full_output_scan",
        )

    if not candidates:
        return _grading_result(
            parsed_answer=parsed_answer,
            correctness=None,
            status="ambiguous",
            reason="no_candidate_extracted",
            usable_for_metrics=False,
        )
    if len(candidates) > 1:
        return _grading_result(
            parsed_answer=parsed_answer,
            correctness=None,
            status="ambiguous",
            reason="multiple_candidates",
            usable_for_metrics=False,
            commitment_kind="ambiguous",
            commitment_source="full_output_scan",
        )

    candidate = candidates[0]
    matching_letters = [
        option_letter
        for option_letter, option_text in option_map.items()
        if _candidate_matches_option_text(candidate, option_text)
    ]
    matching_letters = list(dict.fromkeys(matching_letters))
    if len(matching_letters) > 1:
        return _grading_result(
            parsed_answer=parsed_answer,
            correctness=None,
            status="ambiguous",
            reason="candidate_matches_multiple_options",
            usable_for_metrics=False,
            commitment_kind="ambiguous",
            commitment_source="full_output_scan",
        )
    if len(matching_letters) == 1:
        is_correct = matching_letters[0] == correct_letter
        return _grading_result(
            parsed_answer=parsed_answer,
            correctness=int(is_correct),
            status="correct" if is_correct else "incorrect",
            reason="single_option_text_match" if is_correct else "single_option_text_non_match",
            usable_for_metrics=True,
            committed_answer=candidate,
            commitment_kind="option_text",
            commitment_source="full_output_scan",
        )

    if _candidate_matches_gold(candidate, gold_answers):
        return _grading_result(
            parsed_answer=parsed_answer,
            correctness=1,
            status="correct",
            reason="single_candidate_match",
            usable_for_metrics=True,
            committed_answer=candidate,
            commitment_kind="text",
            commitment_source="full_output_scan",
        )

    return _grading_result(
        parsed_answer=parsed_answer,
        correctness=0,
        status="incorrect",
        reason="single_candidate_non_match",
        usable_for_metrics=True,
        committed_answer=candidate,
        commitment_kind="text",
        commitment_source="full_output_scan",
    )


def grade_multiple_choice_response(
    text: str,
    base: Dict[str, Any],
    generation_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if _multiple_choice_mode(base) in {MC_MODE_STRICT, MC_MODE_WITH_RATIONALE}:
        return _grade_multiple_choice_response_strict(text, base, generation_info=generation_info)
    return _grade_multiple_choice_response_legacy(text, base)


def grade_response_from_base(
    text: str,
    base: Dict[str, Any],
    generation_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if is_multiple_choice_base(base):
        return grade_multiple_choice_response(text, base, generation_info=generation_info)
    return grade_short_answer(text, extract_gold_answers_from_base(base))


def is_correct_short_answer(pred: str, gold_answers: List[str]) -> bool:
    grading = grade_short_answer(pred, gold_answers)
    return bool(grading["usable_for_metrics"] and grading["correctness"] == 1)


__all__ = [
    "extract_gold_answers_from_base",
    "normalize_answer",
    "is_correct_short_answer",
    "extract_short_answer_from_generation",
    "grade_short_answer",
    "grade_multiple_choice_response",
    "grade_response_from_base",
]
