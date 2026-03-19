from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _records_for_template(records: Sequence[Dict[str, Any]], template_type: str) -> List[Dict[str, Any]]:
    return [record for record in records if record.get("template_type") == template_type]


def build_probe_record_sets(
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    all_records: Sequence[Dict[str, Any]],
    bias_types: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    families: Dict[str, Dict[str, Any]] = {
        "neutral": {
            "template_type": "neutral",
            "desc": "no_bias",
            "meta_key": "probe_no_bias",
            "score_key": "probe_x",
            "train_records": _records_for_template(train_records, "neutral"),
            "val_records": _records_for_template(val_records, "neutral"),
            "score_records": _records_for_template(all_records, "neutral"),
        }
    }
    families["neutral"]["retrain_records"] = families["neutral"]["train_records"] + families["neutral"]["val_records"]

    for bias_type in bias_types:
        train_subset = _records_for_template(train_records, bias_type)
        val_subset = _records_for_template(val_records, bias_type)
        families[bias_type] = {
            "template_type": bias_type,
            "desc": f"bias:{bias_type}",
            "meta_key": f"probe_bias_{bias_type}",
            "score_key": "probe_xprime",
            "train_records": train_subset,
            "val_records": val_subset,
            "retrain_records": train_subset + val_subset,
            "score_records": _records_for_template(all_records, bias_type),
        }

    return families


__all__ = ["build_probe_record_sets"]
