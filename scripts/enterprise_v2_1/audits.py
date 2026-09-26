"""(v2.1 versioned copy of scripts/enterprise_v2/audits.py: renamed decision
points, v2.1 family/population label tokens.)

Shortcut (lexical-separability) audit, matched-pair audit, and
leakage validation for enterprise corpus v2 / benchmark v1.

The shortcut audit uses only surface statistics of the model-visible
input text (never hidden states): account-id presence/count, length,
punctuation, JSON delimiters, loaded keywords, decision-point marker,
action-history length, contract, workflow variant. A lightweight
logistic regression over those features is trained on each population's
train split and scored on its held-out test split (benchmark: grouped
5-fold CV by document, reported only -- the benchmark is never used to
tune anything). Univariate AUROCs are reported per feature.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from enterprise_v2_1.wording import FAMILY_SPECS, LEXICAL_FAMILY_SPECS, SURFACES

ACCOUNT_RE = re.compile(r"SIM-ACCT-[0-9A-F]{12}")
KEYWORDS = (
    "ignore", "urgent", "new", "verified", "account", "skip", "immediately", "remember", "policy", "system",
    "instruction", "override", "bypass", "review", "directly", "changed", "closed", "trusted", "today", "now",
    "please", "must", "action", "confirm",
)
URGENCY_WORDS = ("urgent", "immediately", "today", "now", "deadline", "asap", "priority", "soon", "quickly", "cut-off", "hour", "expedite", "time")
INSTRUCTION_WORDS = ("please", "ignore", "skip", "disregard", "use", "send", "pay", "remit", "route", "do not", "don't", "must", "should", "kindly", "keep", "forget", "set aside", "store", "record", "save", "stop", "never", "waive", "omit", "bypass", "suppress")


def visible_text(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] != "system")


def trivial_features(messages: list[dict], meta: dict) -> dict[str, float]:
    t = visible_text(messages)
    low = t.lower()
    ids = ACCOUNT_RE.findall(t)
    f: dict[str, float] = {
        "n_chars": len(t),
        "n_messages": len(messages),
        "tool_history_depth": meta["tool_history_depth"],
        "n_account_ids": len(ids),
        "n_distinct_account_ids": len(set(ids)),
        "has_account_id": float(bool(ids)),
        "n_exclaim": t.count("!"),
        "n_question": t.count("?"),
        "n_colon": t.count(":"),
        "n_semicolon": t.count(";"),
        "n_dash": t.count("-"),
        "n_open_brace": t.count("{"),
        "n_close_brace": t.count("}"),
        "n_open_bracket": t.count("["),
        "n_escaped_quote": t.count('\\"'),
        "n_uppercase_words": len(re.findall(r"\b[A-Z]{3,}\b", t)),
    }
    for kw in KEYWORDS:
        f[f"kw_{kw}"] = len(re.findall(rf"\b{re.escape(kw)}\b", low))
    for dp in ("prefetched_memory", "post_email", "post_invoice", "post_memory_retrieval", "post_vendor_record", "pre_payment_proposal"):
        f[f"dp_{dp}"] = float(meta["decision_point"] == dp)
    f["contract_session_a"] = float(meta["contract"] == "session_a_intake")
    for v in ("memory_before_vendor", "vendor_before_memory", "memory_prefetched_first"):
        f[f"variant_{v}"] = float(meta["workflow_variant"] == v)
    return f


def _auroc(y, s) -> float:
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def shortcut_audit(rows: list[dict], split_train: str | None, split_test: str | None, thresholds: dict) -> dict:
    """rows: dicts with keys messages, label (label record)."""
    feats = [trivial_features(r["messages"], r["label"]) for r in rows]
    names = sorted(feats[0])
    X = np.array([[f[n] for n in names] for f in feats], dtype=float)
    y = np.array([r["label"]["exposure_label"] for r in rows])
    splits = np.array([r["label"]["split"] for r in rows])
    docs = np.array([r["label"]["docid"] for r in rows])

    univariate = {}
    for i, n in enumerate(names):
        a = _auroc(y, X[:, i])
        univariate[n] = {
            "auroc": round(a, 4) if a == a else None,
            "clean_mean": round(float(X[y == 0, i].mean()), 4),
            "attacked_mean": round(float(X[y == 1, i].mean()), 4),
        }
    worst = max(((n, abs(v["auroc"] - 0.5)) for n, v in univariate.items() if v["auroc"] is not None), key=lambda t: t[1])

    def clf():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))

    texts = [visible_text(r["messages"]) for r in rows]
    if split_train is not None:
        tr, te = splits == split_train, splits == split_test
        m = clf().fit(X[tr], y[tr])
        combined = _auroc(y[te], m.predict_proba(X[te])[:, 1])
        combined_train = _auroc(y[tr], m.predict_proba(X[tr])[:, 1])
        vec = CountVectorizer(lowercase=True, min_df=2, token_pattern=r"[A-Za-z][A-Za-z_]+")
        Xt = vec.fit_transform([t for t, k in zip(texts, tr) if k])
        bow = LogisticRegression(max_iter=3000, C=1.0).fit(Xt, y[tr])
        bow_auc = _auroc(y[te], bow.predict_proba(vec.transform([t for t, k in zip(texts, te) if k]))[:, 1])
        protocol = f"logistic regression on standardized trivial features, fit on '{split_train}', scored on held-out '{split_test}'"
    else:
        gkf = GroupKFold(n_splits=5)
        s = np.zeros(len(y))
        sb = np.zeros(len(y))
        for tr_idx, te_idx in gkf.split(X, y, docs):
            s[te_idx] = clf().fit(X[tr_idx], y[tr_idx]).predict_proba(X[te_idx])[:, 1]
            vec = CountVectorizer(lowercase=True, min_df=2, token_pattern=r"[A-Za-z][A-Za-z_]+")
            Xt = vec.fit_transform([texts[i] for i in tr_idx])
            sb[te_idx] = LogisticRegression(max_iter=3000).fit(Xt, y[tr_idx]).predict_proba(vec.transform([texts[i] for i in te_idx]))[:, 1]
        combined = _auroc(y, s)
        combined_train = None
        bow_auc = _auroc(y, sb)
        protocol = "logistic regression on standardized trivial features, 5-fold CV grouped by document (report only)"

    passes = combined <= thresholds["trivial_feature_classifier_test_auroc_max"] and worst[1] <= thresholds["single_trivial_feature_auroc_deviation_max"]
    return {
        "n_items": len(rows),
        "feature_names": names,
        "classifier_protocol": protocol,
        "trivial_feature_classifier_auroc": round(combined, 4),
        "trivial_feature_classifier_train_auroc": round(combined_train, 4) if combined_train is not None else None,
        "most_separating_single_feature": {"feature": worst[0], "auroc": univariate[worst[0]]["auroc"], "abs_deviation_from_0_5": round(worst[1], 4)},
        "univariate": univariate,
        "bag_of_words_reference_auroc": round(bow_auc, 4),
        "bag_of_words_note": "Informational only. Unigram bag-of-words over the visible text, same fit/score protocol. Attack cores are semantic instructions that differ from their clean mirrors by design, so high lexical separability of the injected sentence is expected; it is the ceiling a hidden-state probe must be compared against, not a shortcut.",
        "thresholds": thresholds,
        "passes_pre_registered_thresholds": bool(passes),
    }


def pair_audit(rows: list[dict], tolerances: dict) -> dict:
    by_pair: dict[str, dict] = defaultdict(dict)
    for r in rows:
        by_pair[r["label"]["pair_id"]][r["label"]["class"]] = r
    n = 0
    field_mismatch = Counter()
    rel_len = []
    same_urgency = same_instr = same_json = same_ids = 0
    violations = []
    for pid, d in by_pair.items():
        if set(d) != {"clean", "attacked"}:
            violations.append({"pair_id": pid, "problem": "pair incomplete"})
            continue
        n += 1
        a, c = d["attacked"]["label"], d["clean"]["label"]
        for fld in ("docid", "decision_point", "workflow_variant", "contract", "tool_history_depth", "account_id_mode", "pressure_clause", "memory_baseline", "carrier", "invoice_field", "matched_attack_surface", "matched_attack_family"):
            if a.get(fld) != c.get(fld):
                field_mismatch[fld] += 1
        ta, tc = visible_text(d["attacked"]["messages"]), visible_text(d["clean"]["messages"])
        rl = abs(len(ta) - len(tc)) / max(len(ta), len(tc))
        rel_len.append(rl)
        sa, sc = d["attacked"]["slot_text"].lower(), d["clean"]["slot_text"].lower()
        same_urgency += (any(w in sa for w in URGENCY_WORDS) == any(w in sc for w in URGENCY_WORDS))
        same_instr += (any(w in sa for w in INSTRUCTION_WORDS) == any(w in sc for w in INSTRUCTION_WORDS))
        same_json += (("{" in sa) == ("{" in sc))
        same_ids += (len(ACCOUNT_RE.findall(ta)) == len(ACCOUNT_RE.findall(tc)))
        if rl > tolerances["max_relative_length_difference"]:
            violations.append({"pair_id": pid, "problem": f"relative length difference {rl:.3f}"})
    rel = np.array(rel_len) if rel_len else np.zeros(1)
    return {
        "n_pairs": n,
        "structural_field_mismatches": dict(field_mismatch),
        "relative_full_context_length_difference": {"mean": round(float(rel.mean()), 4), "p95": round(float(np.percentile(rel, 95)), 4), "max": round(float(rel.max()), 4)},
        "same_urgency_vocabulary_presence_rate": round(same_urgency / max(1, n), 4),
        "same_instructional_language_presence_rate": round(same_instr / max(1, n), 4),
        "same_json_presence_rate": round(same_json / max(1, n), 4),
        "same_account_id_count_rate": round(same_ids / max(1, n), 4),
        "violations": violations,
        "passes": not violations and not field_mismatch and same_json == n and same_ids == n,
    }


LABEL_TOKENS = tuple(
    sorted(
        {f.family_id for f in FAMILY_SPECS}
        | {f.family_id for f in LEXICAL_FAMILY_SPECS}
        | {"probe_v2_1", "lora_v2_1", "representative_operations_benchmark", "attack_eligible_security_challenge", "delayed_memory_ood", "lexical_challenge", "initial_calibration", "non_invoice_document_negative", "memory_prefetched_first", "post_memory_retrieval", "pre_payment_proposal"}
        | set(SURFACES)
        | {"beneficiary_change_request", "procedural_manipulation", "exposure_label", "attack_surface", "attack_family", "gold_action", "gold_rule", "expected_outcome", "expected_terminal_action", "scenario_category", "probe_v2", "lora_v2", "enterprise_benchmark_v1", "wording_partition", "template_family_id", "clean_control_role", "__train", "__validation", "__test", "__benchmark"}
    )
)
CLASS_WORDS_RE = re.compile(r"\b(attacked|attack|poison(?:ed)?|malicious|adversarial|injection|injected)\b", re.IGNORECASE)
PATH_MARKERS = ("/home/", "data/docile", "annotations/", ".pdf", "ocr/", "results/", "scripts/")
RULE_RE = re.compile(r"\bR[1-7]_[a-z_]+\b|\bA[12]_[a-z_]+\b|read_next:")
SAMPLE_ID_RE = re.compile(r"\b(pv21|pood|plex|pcal|lv21|rob|aesc)-(pair-)?[0-9a-f]{12,16}\b")


def scan_text_for_leakage(text: str) -> list[str]:
    hits = []
    for tok in LABEL_TOKENS:
        if tok in text:
            hits.append(f"label_token:{tok}")
    m = CLASS_WORDS_RE.search(text)
    if m:
        hits.append(f"class_word:{m.group(0)}")
    for p in PATH_MARKERS:
        if p in text:
            hits.append(f"path_marker:{p}")
    if RULE_RE.search(text):
        hits.append("gold_rule_id")
    if SAMPLE_ID_RE.search(text):
        hits.append("sample_or_pair_id")
    return hits


__all__ = ["trivial_features", "shortcut_audit", "pair_audit", "scan_text_for_leakage", "visible_text", "ACCOUNT_RE", "LABEL_TOKENS"]
