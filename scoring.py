"""Deterministic vendor scoring. Every number shown in the app comes from here, never from the AI."""
from __future__ import annotations

import io
import re

import pandas as pd

REQUIRED_COLUMNS = {
    "vendor_id": "text",
    "vendor_name": "text",
    "category": "text",
    "city": "text",
    "unit": "text",
    "unit_price_inr": "number",
    "quality_score": "number",
    "defect_rate_pct": "number",
    "on_time_delivery_pct": "number",
    "lead_time_days": "number",
    "moq": "number",
    "payment_terms_days": "number",
    "iso_certified": "yes/no",
    "years_in_business": "number",
    "disputes_last_2y": "number",
    "notes": "text",
}

# Allowed range for each numeric column (used to reject bad uploads).
RANGES = {
    "unit_price_inr": (0.01, 10_000_000),
    "quality_score": (0, 10),
    "defect_rate_pct": (0, 100),
    "on_time_delivery_pct": (0, 100),
    "lead_time_days": (0, 365),
    "moq": (0, 10_000_000),
    "payment_terms_days": (0, 365),
    "years_in_business": (0, 200),
    "disputes_last_2y": (0, 1000),
}

CRITERIA = {
    "cost": "Cost",
    "quality": "Quality",
    "speed": "Lead time",
    "reliability": "Reliability",
    "payment": "Payment terms",
}

PRESETS = {
    "Balanced": {"cost": 35, "quality": 30, "speed": 15, "reliability": 15, "payment": 5},
    "Cost-first": {"cost": 60, "quality": 20, "speed": 5, "reliability": 10, "payment": 5},
    "Quality-first": {"cost": 20, "quality": 50, "speed": 5, "reliability": 20, "payment": 5},
    "Urgent order": {"cost": 20, "quality": 20, "speed": 40, "reliability": 20, "payment": 0},
}
DEFAULT_WEIGHTS = PRESETS["Balanced"]

# Smallest range used when scaling a metric. Without it, a Rs 0.10 price gap between two
# vendors would become a full 0-vs-100 swing on cost (found while testing, see report D-Q1).
MIN_RANGE = {
    "unit_price_inr": None,  # set to 10% of the median price at run time
    "quality_score": 1.0,
    "defect_rate_pct": 0.5,
    "lead_time_days": 2.0,
    "on_time_delivery_pct": 5.0,
    "disputes_last_2y": 1.0,
    "payment_terms_days": 15.0,
}

MASK_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b"), "[GSTIN]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[card/account no.]"),
    (re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b"), "[phone]"),
]


def mask_pii(text: str) -> str:
    """Hide e-mails, phone numbers, PAN, GSTIN and card/account numbers before text goes to the AI."""
    if not isinstance(text, str):
        return text
    for pattern, label in MASK_PATTERNS:
        text = pattern.sub(label, text)
    return text


# ---------------------------------------------------------------- data loading
def load_csv(file) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Read and validate a vendor CSV. Returns (dataframe or None, errors, warnings)."""
    errors, warnings = [], []
    try:
        raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig", errors="replace")
        df = pd.read_csv(io.StringIO(raw))
    except Exception as exc:  # noqa: BLE001
        return None, [f"Could not read the file as CSV ({exc.__class__.__name__})."], []

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, [f"Missing column(s): {', '.join(missing)}. Download the template to see the format."], []
    df = df[list(REQUIRED_COLUMNS)].copy()

    if len(df) == 0:
        return None, ["The file has no vendor rows."], []
    if len(df) > 2000:
        return None, ["Please keep the file under 2,000 vendors."], []

    for col in ["vendor_id", "vendor_name", "category", "city", "unit", "notes"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    blank = df[(df.vendor_id == "") | (df.vendor_name == "") | (df.category == "")]
    if len(blank):
        warnings.append(f"Dropped {len(blank)} row(s) with no vendor ID, name or category.")
        df = df.drop(blank.index)

    dupes = df[df.duplicated("vendor_id", keep="first")]
    if len(dupes):
        warnings.append(f"Dropped {len(dupes)} duplicate vendor ID(s): {', '.join(dupes.vendor_id.head(5))}.")
        df = df.drop(dupes.index)

    for col, (lo, hi) in RANGES.items():
        cleaned = pd.to_numeric(
            df[col].astype(str).str.replace(",", "").str.replace("₹", "").str.replace("%", "").str.strip(),
            errors="coerce",
        )
        bad = df[cleaned.isna() | (cleaned < lo) | (cleaned > hi)]
        if len(bad):
            warnings.append(
                f"Dropped {len(bad)} row(s) where '{col}' was missing or outside {lo:g} to {hi:g} "
                f"(e.g. {', '.join(bad.vendor_id.head(3))})."
            )
            df = df.drop(bad.index)
            cleaned = cleaned.drop(bad.index)
        df[col] = cleaned

    df["iso_certified"] = (
        df["iso_certified"].astype(str).str.strip().str.lower().isin(["yes", "y", "true", "1"]).map({True: "Yes", False: "No"})
    )
    df["notes"] = df["notes"].map(mask_pii).str.slice(0, 200)
    df["vendor_name"] = df["vendor_name"].str.slice(0, 80)

    if len(df) == 0:
        errors.append("No valid vendor rows remain after checking the data.")
        return None, errors, warnings
    return df.reset_index(drop=True), errors, warnings


def template_csv() -> str:
    return pd.read_csv("data/vendors.csv").head(3).to_csv(index=False)


# ---------------------------------------------------------------- scoring
def normalise_weights(weights: dict) -> dict:
    total = sum(max(0, v) for v in weights.values())
    if total <= 0:
        return {}
    return {k: max(0, v) / total for k, v in weights.items()}


def _scale(series: pd.Series, ref: pd.Series, higher_is_better: bool, min_range: float) -> pd.Series:
    lo, hi = ref.min(), ref.max()
    rng = max(hi - lo, min_range)
    if rng == 0:
        return pd.Series(1.0, index=series.index)
    out = (series - lo) / rng if higher_is_better else (hi - series) / rng
    return out.clip(0, 1)


def filter_vendors(df: pd.DataFrame, req: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the hard constraints. Returns (eligible, excluded-with-reason)."""
    pool = df[df.category == req["category"]].copy()
    reasons = []
    for _, r in pool.iterrows():
        why = []
        if req["quantity"] < r.moq:
            why.append(f"order qty {req['quantity']:,} below MOQ {int(r.moq):,}")
        if req.get("max_lead_days") and r.lead_time_days > req["max_lead_days"]:
            why.append(f"lead time {int(r.lead_time_days)} d > {req['max_lead_days']} d allowed")
        if req.get("iso_required") and r.iso_certified != "Yes":
            why.append("not ISO certified")
        if req.get("max_unit_price") and r.unit_price_inr > req["max_unit_price"]:
            why.append(f"price ₹{r.unit_price_inr:,.2f} above cap ₹{req['max_unit_price']:,.2f}")
        reasons.append("; ".join(why))
    pool["exclusion_reason"] = reasons
    return pool[pool.exclusion_reason == ""].drop(columns="exclusion_reason"), pool[pool.exclusion_reason != ""]


def score(eligible: pd.DataFrame, weights: dict, quantity: int, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score eligible vendors 0-100 and rank them.

    `reference` = every vendor in the category (eligible or not). Scaling against it means a filter that removes a
    vendor cannot reshuffle the others (rank reversal found in testing; see report D-Q1).
    """
    if eligible.empty:
        return eligible
    w = normalise_weights(weights)
    d = eligible.copy()
    ref = reference if reference is not None and not reference.empty else eligible
    price_floor = 0.10 * ref.unit_price_inr.median()
    m = lambda col: MIN_RANGE[col] if MIN_RANGE[col] is not None else price_floor  # noqa: E731
    sc_ = lambda col, hib: _scale(d[col], ref[col], hib, m(col))  # noqa: E731

    d["s_cost"] = sc_("unit_price_inr", False)
    d["s_quality"] = 0.5 * sc_("quality_score", True) + 0.5 * sc_("defect_rate_pct", False)
    d["s_speed"] = sc_("lead_time_days", False)
    d["s_reliability"] = 0.7 * sc_("on_time_delivery_pct", True) + 0.3 * sc_("disputes_last_2y", False)
    d["s_payment"] = sc_("payment_terms_days", True)

    for k in CRITERIA:
        d[f"pts_{k}"] = 100 * w.get(k, 0) * d[f"s_{k}"]
    d["score"] = sum(d[f"pts_{k}"] for k in CRITERIA).round(1)
    d["order_value_inr"] = (d.unit_price_inr * quantity).round(0)
    d = d.sort_values(["score", "unit_price_inr"], ascending=[False, True]).reset_index(drop=True)
    d.insert(0, "rank", range(1, len(d) + 1))
    return d


def sensitivity(eligible: pd.DataFrame, weights: dict, quantity: int, step: int = 10, reference=None) -> dict:
    """Nudge each weight up/down by `step` points and see whether the winner changes."""
    base = score(eligible, weights, quantity, reference)
    if len(base) < 2:
        return {"scenarios": 0, "same_winner": 0, "flips": []}
    winner = base.iloc[0].vendor_id
    scenarios, same, flips = 0, 0, []
    for k in CRITERIA:
        for delta in (+step, -step):
            w2 = dict(weights)
            w2[k] = max(0, w2.get(k, 0) + delta)
            if sum(w2.values()) == 0:
                continue
            top = score(eligible, w2, quantity, reference).iloc[0]
            scenarios += 1
            if top.vendor_id == winner:
                same += 1
            else:
                flips.append(f"{CRITERIA[k]} {'+' if delta > 0 else '−'}{step} → {top.vendor_name}")
    return {"scenarios": scenarios, "same_winner": same, "flips": flips}


def rule_checks(ranked: pd.DataFrame, quantity: int) -> list[tuple[str, str]]:
    """Simple rules a purchase manager would apply. Returns (level, message) pairs."""
    if ranked.empty:
        return [("error", "No vendor meets the requirement. Relax the lead time, price cap or ISO filter.")]
    top = ranked.iloc[0]
    out = []
    if len(ranked) == 1:
        out.append(("warning", "Only one vendor qualifies, so there is no competition and you are single-sourcing."))
    l1 = ranked.sort_values("unit_price_inr").iloc[0]
    if l1.vendor_id == top.vendor_id:
        out.append(("success", f"The top-ranked vendor is also L1, the cheapest eligible vendor at ₹{top.unit_price_inr:,.2f}/{top.unit}."))
    else:
        prem = (top.unit_price_inr / l1.unit_price_inr - 1) * 100
        extra = (top.unit_price_inr - l1.unit_price_inr) * quantity
        out.append(
            ("info", f"The top-ranked vendor costs {prem:.1f}% more than L1 ({l1.vendor_name}, ₹{l1.unit_price_inr:,.2f}): "
                     f"₹{extra:,.0f} extra on this order. Check that the quality and delivery gain is worth it.")
        )
    if top.defect_rate_pct > 2:
        out.append(("warning", f"The top vendor's defect rate is {top.defect_rate_pct}%, above the 2% rule of thumb."))
    if top.on_time_delivery_pct < 85:
        out.append(("warning", f"The top vendor's on-time delivery is {top.on_time_delivery_pct:.0f}%, below the 85% rule of thumb."))
    if top.disputes_last_2y >= 2:
        out.append(("warning", f"The top vendor has {int(top.disputes_last_2y)} disputes in the last 2 years."))
    if len(ranked) > 1:
        gap = top.score - ranked.iloc[1].score
        if gap < 3:
            out.append(("warning", f"Close call: only {gap:.1f} points separate #1 and #2 ({ranked.iloc[1].vendor_name}). "
                                   "Consider splitting the order 70/30 or negotiating with both."))
    return out


def build_payload(req: dict, weights: dict, ranked: pd.DataFrame, excluded: pd.DataFrame, sens: dict, checks) -> dict:
    """Compact, pre-computed facts handed to the AI. The AI only explains these numbers."""
    w = normalise_weights(weights)
    cols = ["rank", "vendor_id", "vendor_name", "city", "unit_price_inr", "order_value_inr", "quality_score",
            "defect_rate_pct", "on_time_delivery_pct", "lead_time_days", "moq", "payment_terms_days",
            "iso_certified", "years_in_business", "disputes_last_2y", "score", "notes"]
    top = ranked.head(6)
    vendors = []
    for _, r in top.iterrows():
        v = {c: (r[c].item() if hasattr(r[c], "item") else r[c]) for c in cols}
        v["points_by_criterion"] = {CRITERIA[k]: round(float(r[f"pts_{k}"]), 1) for k in CRITERIA}
        vendors.append(v)
    facts = {}
    if len(ranked):
        t = ranked.iloc[0]
        l1 = ranked.sort_values("unit_price_inr").iloc[0]
        facts = {
            "rank1_vendor_id": t.vendor_id,
            "l1_vendor_id": l1.vendor_id,
            "rank1_premium_over_l1_pct": round((t.unit_price_inr / l1.unit_price_inr - 1) * 100, 1),
            "rank1_extra_cost_vs_l1_inr": round(float((t.unit_price_inr - l1.unit_price_inr) * req["quantity"]), 0),
            "gap_rank1_vs_rank2_points": round(float(t.score - ranked.iloc[1].score), 1) if len(ranked) > 1 else None,
            "winner_stable_in": f"{sens['same_winner']} of {sens['scenarios']} weight changes",
        }
    return {
        "requirement": {
            "category": req["category"], "quantity": req["quantity"], "unit": ranked.iloc[0].unit if len(ranked) else "",
            "max_lead_days": req.get("max_lead_days"), "iso_required": req.get("iso_required"),
            "max_unit_price_inr": req.get("max_unit_price") or None,
        },
        "weights_pct": {CRITERIA[k]: round(v * 100) for k, v in w.items()},
        "ranked_vendors": vendors,
        "excluded_vendors": [
            {"vendor_name": r.vendor_name, "reason": r.exclusion_reason} for _, r in excluded.iterrows()
        ],
        "precomputed_facts": facts,
        "rule_checks": [m for _, m in checks],
    }
