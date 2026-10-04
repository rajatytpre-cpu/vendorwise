"""VendorWise: AI-assisted vendor selection (AI for Managers end-term project, Use Case #6)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import ai
import scoring as sc

st.set_page_config(page_title="VendorWise · AI Vendor Selection", page_icon="📦", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 3.2rem; max-width: 1250px;}
.vw-kpi {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:12px 16px; height:100%;}
.vw-kpi .l {font-size:.8rem; opacity:.7;} .vw-kpi .v {font-size:1.35rem; font-weight:700; line-height:1.3;} .vw-kpi .s {font-size:.8rem; opacity:.8;}
.vw-hero {background: linear-gradient(120deg,#0f3d3e 0%,#14625f 60%,#1f7a6e 100%); color:#fff; padding:18px 24px;
          border-radius:14px; margin-bottom:14px;}
.vw-hero h1 {color:#fff; font-size:1.75rem; margin:0 0 2px 0; padding:0;}
.vw-hero p {color:#d7efe9; margin:0; font-size:0.95rem;}
.vw-pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; margin-right:6px;}
.vw-ok {background:#dcfce7; color:#14532d;} .vw-warn {background:#fef3c7; color:#78350f;} .vw-bad {background:#fee2e2; color:#7f1d1d;}
.vw-card {border:1px solid rgba(128,128,128,.25); border-radius:12px; padding:14px 18px; margin-bottom:10px;}
.vw-small {font-size:0.82rem; opacity:.75;}
</style>
""",
    unsafe_allow_html=True,
)

def inr(x: float, dec: int = 0) -> str:
    """Indian digit grouping: 344000 -> ₹3,44,000."""
    neg, x = x < 0, abs(float(x))
    whole, frac = f"{x:.{dec}f}".split(".") if dec else (f"{x:.0f}", "")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:]); head = head[:-2]
    if head:
        groups.insert(0, head)
    out = ",".join(groups + [tail]) if groups else tail
    return ("-" if neg else "") + "₹" + out + (f".{frac}" if frac else "")


def kpi(col, label, value, sub="", tone=""):
    color = {"good": "#15803d", "bad": "#b45309"}.get(tone, "inherit")
    col.markdown(f'<div class="vw-kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s" style="color:{color}">{sub}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ state
ss = st.session_state
defaults = {
    "df": None, "data_label": "Sample data · Shakti Appliances Pvt. Ltd. (fictional)",
    "memo": None, "memo_hash": None, "chat": [], "ai_calls": 0, "parse_msg": None,
    "quantity": 5000, "max_lead": 14, "iso": False, "max_price": 0.0, "user_key": "",
}
for k, v in defaults.items():
    ss.setdefault(k, v)
for k, v in sc.DEFAULT_WEIGHTS.items():
    ss.setdefault(f"w_{k}", v)


@st.cache_data(show_spinner=False)
def sample_data():
    df, _, _ = sc.load_csv("data/vendors.csv")
    return df


if ss.df is None:
    ss.df = sample_data()
df: pd.DataFrame = ss.df
categories = sorted(df.category.unique())
if ss.get("category") not in categories:
    ss.category = categories[0]


# ------------------------------------------------------------------ AI setup
def secret_key() -> str:
    try:
        return st.secrets.get("GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    except Exception:  # no secrets file
        return os.environ.get("GEMINI_API_KEY", "")


def models_from_secrets() -> list[str]:
    try:
        m = st.secrets.get("GEMINI_MODEL", "")
    except Exception:
        m = os.environ.get("GEMINI_MODEL", "")
    return ([m] if m else []) + [x for x in ai.DEFAULT_MODELS if x != m]


@st.cache_resource(show_spinner=False)
def get_client(key: str, models: tuple):
    return ai.GeminiClient(key, list(models))


@st.cache_data(ttl=600, show_spinner=False)
def check_ai(key: str, models: tuple, nonce: int):
    try:
        model = get_client(key, models).ping()
        return {"ok": True, "model": model, "msg": ""}
    except ai.AIError as e:
        return {"ok": False, "model": None, "msg": e.message, "kind": e.kind}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "model": None, "msg": f"AI setup failed ({e.__class__.__name__}).", "kind": "network"}


@st.cache_data(ttl=3600, show_spinner=False)
def cached_memo(key: str, models: tuple, payload_json: str):
    """Same inputs → same memo, shared across users for an hour (saves free-tier quota)."""
    return get_client(key, models).write_memo(json.loads(payload_json))


API_KEY = ss.user_key.strip() or secret_key()
MODELS = tuple(models_from_secrets())
ss.setdefault("ai_nonce", 0)
status = check_ai(API_KEY, MODELS, ss.ai_nonce) if API_KEY else {"ok": False, "msg": "No Gemini API key configured.", "kind": "key"}
if ss.ai_calls >= ai.MAX_CALLS_PER_SESSION:
    status = {"ok": False, "msg": f"Session limit of {ai.MAX_CALLS_PER_SESSION} AI calls reached (protects the free quota).", "kind": "quota"}
AI_ON = status["ok"]


def ai_failed(e: ai.AIError):
    st.warning(f"AI unavailable: {e.message} Showing the rule-based version instead.")
    if e.kind in ("key", "quota", "network"):
        check_ai.clear()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 📦 VendorWise")
    if AI_ON:
        st.markdown(f'<span class="vw-pill vw-ok">● AI mode</span> Gemini `{status["model"]}`', unsafe_allow_html=True)
    else:
        st.markdown('<span class="vw-pill vw-warn">● Basic mode (no AI)</span>', unsafe_allow_html=True)
        st.caption(status["msg"] + " Scoring and the shortlist still work; memos use templates.")
        if st.button("Retry AI mode"):
            ss.ai_nonce += 1
            check_ai.clear()
            st.rerun()
    calls_slot = st.empty()  # filled at the end of the run, after any AI call

    st.divider()
    st.markdown("**Vendor data**")
    src = st.radio("Source", ["Sample data", "Upload my CSV"], label_visibility="collapsed")
    if src == "Upload my CSV":
        up = st.file_uploader("Vendor CSV", type=["csv"], help="Max 2,000 rows. Same columns as the template.")
        st.download_button("Download CSV template", sc.template_csv(), "vendorwise_template.csv", "text/csv")
        if up is not None and st.button("Load this file", type="primary"):
            new, errs, warns = sc.load_csv(up)
            for w in warns:
                st.warning(w)
            if errs:
                for e in errs:
                    st.error(e)
            else:
                ss.df, ss.data_label = new, f"Uploaded: {up.name} ({len(new)} vendors)"
                ss.memo, ss.chat = None, []
                ss.upload_warnings = warns  # kept so they survive the rerun below
                st.rerun()
    elif not ss.data_label.startswith("Sample"):
        ss.df, ss.data_label = sample_data(), defaults["data_label"]
        ss.memo, ss.chat, ss.upload_warnings = None, [], []
        st.rerun()
    st.caption(ss.data_label)
    if not ss.data_label.startswith("Sample"):
        for w in ss.get("upload_warnings", []):
            st.warning(w)

    st.divider()
    with st.expander("🔒 Privacy & AI disclosure"):
        st.markdown(
            "- This app uses **Google Gemini** (a third-party AI). The shortlist data and your typed text are "
            "sent to Google's API to write explanations.\n"
            "- E-mails, phone numbers, PAN, GSTIN and card numbers are **masked** before sending.\n"
            "- Rankings and numbers are calculated by the app, **not** by the AI.\n"
            "- Nothing is stored after you close the tab. Free-tier Gemini may use inputs to improve Google's "
            "products, so do not upload confidential contracts."
        )
    with st.expander("Use your own Gemini key (optional)"):
        st.text_input("Gemini API key", type="password", key="user_key",
                      help="Only used in this browser session. Get one free at aistudio.google.com")
    if st.button("Reset session"):
        for k in list(ss.keys()):
            del ss[k]
        st.rerun()

# ------------------------------------------------------------------ header
st.markdown(
    """<div class="vw-hero"><h1>VendorWise: AI Vendor Selection Assistant</h1>
<p>Score vendors on cost, quality, lead time, reliability and payment terms. Set your own weights,
get a ranked shortlist, and read an AI-written rationale you can check against the numbers.</p></div>""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ callbacks
def apply_preset(name: str):
    for k, v in sc.PRESETS[name].items():
        ss[f"w_{k}"] = v


def parse_request():
    text = ss.get("req_text", "").strip()
    ss.parse_msg = None
    if not text:
        ss.parse_msg = ("warning", "Type what you need to buy first.")
        return
    if ai.looks_like_injection(text):
        ss.parse_msg = ("error", "That looks like an attempt to change my instructions, so I've ignored it. "
                                 "Describe what you need to buy, e.g. '8,000 power cords within 10 days, ISO only'.")
        return
    parsed, how = None, "AI"
    if AI_ON:
        try:
            ss.ai_calls += 1
            parsed = get_client(API_KEY, MODELS).parse_requirement(text, categories)
        except ai.AIError as e:
            ss.parse_msg = ("warning", f"AI unavailable ({e.message}). Used keyword rules instead.")
    if parsed is None:
        parsed, how = ai.fallback_parse(text, categories), "keyword rules"
    if parsed["category"] is None:
        off = parsed.get("understood") == "off-topic"
        ss.parse_msg = ("error", "That doesn't look like a purchase requirement. I can only help shortlist vendors."
                        if off else "I couldn't tell which category you mean. Please choose it in the form below.")
        return
    ss.category = parsed["category"]
    if parsed["quantity"]:
        ss.quantity = parsed["quantity"]
    if parsed["max_lead_days"]:
        ss.max_lead = min(parsed["max_lead_days"], 30)
    ss.iso = parsed["iso_required"]
    ss.max_price = float(parsed["max_unit_price"] or 0.0)
    apply_preset(parsed["priority"])
    note = f"Filled by {how}: {parsed['understood']} Priority: **{parsed['priority']}**."
    if parsed["missing"]:
        note += f" Not specified (kept defaults): {', '.join(parsed['missing'])}."
    ss.parse_msg = ("success", note)


# ------------------------------------------------------------------ tabs
t1, t2, t3, t4, t5 = st.tabs(["① Requirement & weights", "② Ranked shortlist", "③ AI recommendation",
                              "④ Ask VendorWise", "Vendor database"])

with t1:
    st.markdown("##### Describe your need (optional)")
    c1, c2 = st.columns([5, 1])
    c1.text_area("Plain-English requirement", key="req_text", height=70, max_chars=ai.MAX_INPUT_CHARS,
                 placeholder="e.g. Need 10,000 mixer jar bodies within 12 days, ISO certified only, quality matters most",
                 label_visibility="collapsed")
    c2.button("✨ Fill form", on_click=parse_request, use_container_width=True, type="primary")
    if ss.parse_msg:
        getattr(st, ss.parse_msg[0])(ss.parse_msg[1])

    st.markdown("##### Requirement")
    a, b, c, d = st.columns([2.2, 1, 1, 1])
    a.selectbox("Category", categories, key="category")
    unit = df[df.category == ss.category].unit.iloc[0]
    b.number_input(f"Quantity ({unit}s)", min_value=1, max_value=10_000_000, step=100, key="quantity")
    c.slider("Max lead time (days)", 1, 30, key="max_lead")
    d.number_input("Max price / unit ₹ (0 = no cap)", min_value=0.0, step=1.0, key="max_price")
    st.checkbox("ISO-certified vendors only", key="iso")

    st.markdown("##### How much does each factor matter?")
    pc = st.columns(len(sc.PRESETS))
    for col, name in zip(pc, sc.PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), use_container_width=True)
    wc = st.columns(5)
    for col, (k, label) in zip(wc, sc.CRITERIA.items()):
        col.slider(label, 0, 100, key=f"w_{k}", step=5)
    weights = {k: ss[f"w_{k}"] for k in sc.CRITERIA}
    norm = sc.normalise_weights(weights)
    if not norm:
        st.error("All weights are zero. Set at least one factor above 0 to rank vendors.")
    else:
        st.caption("Effective weights: " + " · ".join(f"{sc.CRITERIA[k]} **{v*100:.0f}%**" for k, v in norm.items()))
    with st.expander("How the score is calculated"):
        st.markdown(
            "Each factor is scaled 0–1 across all vendors in the category (best = 1), then multiplied by its weight; "
            "total = 0–100.\n\n"
            "- **Cost:** unit price (lower is better)\n"
            "- **Quality:** 50% audit quality score + 50% defect/damage rate\n"
            "- **Lead time:** days to deliver (lower is better)\n"
            "- **Reliability:** 70% on-time delivery + 30% disputes in last 2 years\n"
            "- **Payment terms:** credit days (higher is better)\n\n"
            "A minimum range is used for each factor so that a tiny gap (e.g. ₹0.10 on a ₹45 box) isn't blown up "
            "into a full 0-vs-100 difference. Vendors that fail a hard filter (MOQ, lead time, ISO, price cap) are "
            "excluded before scoring."
        )

# ------------------------------------------------------------------ compute (deterministic)
req = {"category": ss.category, "quantity": int(ss.quantity), "max_lead_days": int(ss.max_lead),
       "iso_required": bool(ss.iso), "max_unit_price": float(ss.max_price)}
eligible, excluded = sc.filter_vendors(df, req)
pool = df[df.category == req["category"]]
ranked = sc.score(eligible, weights, req["quantity"], pool) if norm else eligible.iloc[0:0]
sens = sc.sensitivity(eligible, weights, req["quantity"], reference=pool) if norm and len(eligible) > 1 else {"scenarios": 0, "same_winner": 0, "flips": []}
checks = sc.rule_checks(ranked, req["quantity"], excluded) if norm else []
payload = sc.build_payload(req, weights, ranked, excluded, sens, checks) if len(ranked) else None
payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str) if payload else ""
phash = hashlib.sha1(payload_json.encode()).hexdigest()[:10] if payload else None

with t2:
    if not norm:
        st.error("Set at least one weight above 0 in tab ①.")
    elif ranked.empty:
        st.error("No vendor meets this requirement.")
        if len(excluded):
            st.dataframe(excluded[["vendor_name", "exclusion_reason"]].rename(
                columns={"vendor_name": "Vendor", "exclusion_reason": "Why excluded"}), hide_index=True, use_container_width=True)
    else:
        top = ranked.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        kpi(m1, "Top-ranked vendor", top.vendor_name, f"{top.score} / 100 points", "good")
        kpi(m2, "Order value (top vendor)", inr(top.order_value_inr), f"{inr(top.unit_price_inr, 2)} per {top.unit}")
        gap = top.score - ranked.iloc[1].score if len(ranked) > 1 else None
        kpi(m3, "Lead over #2", f"{gap:.1f} pts" if gap is not None else "n/a",
            ("close call" if gap < 3 else "clear lead") if gap is not None else "only one vendor",
            "bad" if gap is None or gap < 3 else "good")
        if sens["scenarios"]:
            stable = sens["same_winner"] / sens["scenarios"]
            kpi(m4, "Winner stability", f"{sens['same_winner']}/{sens['scenarios']} weight tests",
                "stable" if stable >= 0.8 else "sensitive to your weights", "good" if stable >= 0.8 else "bad")
        else:
            kpi(m4, "Winner stability", "n/a", "needs 2+ vendors")
        st.write("")
        st.caption(f"{len(ranked)} eligible · {len(excluded)} excluded · {req['quantity']:,} {top.unit}s of {req['category']}")

        st.markdown("##### Sanity checks (procurement rules of thumb)")
        for level, msg in checks:
            getattr(st, level)(msg)
        if sens["flips"]:
            with st.expander(f"Weight changes that would change the winner ({len(sens['flips'])})"):
                st.markdown("\n".join(f"- {f}" for f in sens["flips"]))
                st.caption("Each weight was moved ±10 points while the others stayed the same.")

        st.markdown("##### Ranked shortlist")
        show = ranked.assign(order_value_inr=ranked.order_value_inr.map(inr))[["rank", "vendor_name", "city", "unit_price_inr", "order_value_inr", "quality_score", "defect_rate_pct",
                       "on_time_delivery_pct", "lead_time_days", "payment_terms_days", "iso_certified", "score"]]
        st.dataframe(
            show, hide_index=True, use_container_width=True,
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "vendor_name": "Vendor", "city": "City",
                "unit_price_inr": st.column_config.NumberColumn("₹ / unit", format="₹%.2f"),
                "order_value_inr": st.column_config.TextColumn("Order value"),
                "quality_score": st.column_config.NumberColumn("Quality /10"),
                "defect_rate_pct": st.column_config.NumberColumn("Defect %"),
                "on_time_delivery_pct": st.column_config.NumberColumn("On-time %"),
                "lead_time_days": st.column_config.NumberColumn("Lead (d)"),
                "payment_terms_days": st.column_config.NumberColumn("Credit (d)"),
                "iso_certified": "ISO",
                "score": st.column_config.NumberColumn("Score /100", format="%.1f"),
            },
        )

        st.markdown("##### Where each vendor's points come from")
        long = ranked.head(6).melt(id_vars=["rank", "vendor_name"], value_vars=[f"pts_{k}" for k in sc.CRITERIA],
                                   var_name="criterion", value_name="points")
        long["criterion"] = long.criterion.str.replace("pts_", "").map(sc.CRITERIA)
        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                y=alt.Y("vendor_name:N", sort=list(ranked.head(6).vendor_name), title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
                x=alt.X("sum(points):Q", title="Score (0–100)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("criterion:N", sort=list(sc.CRITERIA.values()),
                                scale=alt.Scale(domain=list(sc.CRITERIA.values()),
                                                range=["#14625f", "#3b82f6", "#f59e0b", "#8b5cf6", "#94a3b8"]),
                                legend=alt.Legend(orient="bottom", title=None)),
                order=alt.Order("criterion_order:Q"),
                tooltip=["vendor_name", "criterion", alt.Tooltip("points:Q", format=".1f")],
            )
            .transform_calculate(criterion_order=f"indexof({json.dumps(list(sc.CRITERIA.values()))}, datum.criterion)")
            .properties(height=48 * min(6, len(ranked)) + 30)
        )
        st.altair_chart(chart, use_container_width=True)

        if len(excluded):
            with st.expander(f"Excluded vendors ({len(excluded)})"):
                st.dataframe(excluded[["vendor_name", "city", "unit_price_inr", "exclusion_reason"]].rename(
                    columns={"vendor_name": "Vendor", "city": "City", "unit_price_inr": "₹ / unit",
                             "exclusion_reason": "Why excluded"}), hide_index=True, use_container_width=True)
        st.download_button("⬇ Download shortlist (CSV)", ranked.drop(columns=[c for c in ranked if c.startswith("s_")]).to_csv(index=False),
                           f"shortlist_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")


# ------------------------------------------------------------------ memo tab
def render_memo(m: dict, payload: dict):
    names = {v["vendor_id"]: v["vendor_name"] for v in payload["ranked_vendors"]}
    conf = m.get("confidence", "medium")
    pill = {"high": "vw-ok", "medium": "vw-warn", "low": "vw-bad"}[conf]
    src = "Rule-based summary (AI off)" if m.get("basic_mode") else f"Written by Gemini · {status.get('model') or ''}"
    checks_html = "" if m.get("basic_mode") else (
        ('<span class="vw-pill vw-ok">✓ AI agrees with the scoring</span>' if m.get("agreed_with_scoring")
         else '<span class="vw-pill vw-bad">AI disagreed with the scoring</span>')
        + ('<span class="vw-pill vw-ok">✓ All figures match the data</span>' if not m.get("unverified")
           else f'<span class="vw-pill vw-bad">⚠ {len(m["unverified"])} figure(s) not found in data</span>'))
    st.markdown(
        f'<span class="vw-pill {pill}">Confidence: {conf}</span>' + checks_html + f'<span class="vw-small">{src}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f"### {m['headline']}")
    st.markdown(f"**Recommended:** {names.get(m['recommended_vendor_id'], m['recommended_vendor_id'])}"
                + (f" &nbsp;·&nbsp; **Backup:** {names[m['backup_vendor_id']]}" if m.get("backup_vendor_id") else ""))
    if not m.get("agreed_with_scoring"):
        st.error(f"The AI suggested **{m.get('ai_suggested')}**, which is not the rank-1 vendor. "
                 "The app keeps the scored ranking. Read the AI's reasons critically.")
    if m.get("unverified"):
        st.warning("These figures in the memo could not be traced to the vendor data: "
                   + ", ".join(m["unverified"]) + ". Check them before using the memo.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Why this vendor**\n" + "\n".join(f"- {x}" for x in m["why"]))
        st.markdown("**Trade-offs**\n" + "\n".join(f"- {x}" for x in m["trade_offs"]))
    with c2:
        st.markdown("**Risks to watch**\n" + "\n".join(f"- {x}" for x in m["risks"] or ["None flagged in the data."]))
        st.markdown("**Negotiation points**\n" + "\n".join(f"- {x}" for x in m["negotiation_points"]))
    if m.get("concern"):
        st.info(f"**Second look:** {m['concern']}")
    st.caption("This is decision support, not a decision. Verify vendor documents, samples and current prices before "
               "raising a PO; the buyer remains accountable.")
    md = (f"# VendorWise recommendation\n\nGenerated {datetime.now():%d %b %Y %H:%M}\n\n"
          f"Requirement: {payload['requirement']}\n\nWeights: {payload['weights_pct']}\n\n## {m['headline']}\n\n"
          + "\n".join(f"- {x}" for x in m["why"] + m["trade_offs"] + m["risks"] + m["negotiation_points"]))
    st.download_button("⬇ Download memo (.md)", md, "vendorwise_memo.md", "text/markdown")


with t3:
    if payload is None:
        st.info("Complete tab ① so at least one vendor qualifies.")
    else:
        st.markdown(f"Shortlist for **{req['quantity']:,} × {req['category']}**, top vendor **{ranked.iloc[0].vendor_name}**.")
        go = st.button("✨ Generate recommendation memo" if AI_ON else "Generate rule-based summary", type="primary")
        if go:
            memo = None
            if AI_ON:
                with st.spinner("Gemini is writing the memo from the scored shortlist…"):
                    try:
                        memo = cached_memo(API_KEY, MODELS, payload_json)
                        ss.ai_calls += 1
                    except ai.AIError as e:
                        ai_failed(e)
            if memo is None:
                memo = ai.fallback_memo(payload)
            ss.memo, ss.memo_hash = memo, phash
        if ss.memo:
            if ss.memo_hash != phash:
                st.warning("Your inputs changed after this memo was written. Click the button again to refresh it.")
            render_memo(ss.memo, payload)


# ------------------------------------------------------------------ chat tab
with t4:
    st.caption("Ask about the current shortlist, e.g. *Why not the cheapest vendor?* · *Compare #1 and #2 on delivery* · "
               "*What should I negotiate?* VendorWise is an AI assistant, not a person. For a final decision, "
               "escalate to the purchase manager.")
    box = st.container(height=420)
    with box:
        if not ss.chat:
            st.markdown("👋 Hi, I'm **VendorWise**, an AI assistant. I can explain the ranking, compare vendors and "
                        "suggest negotiation points for the shortlist in tab ②.")
        for msg in ss.chat:
            with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "📦"):
                st.markdown(msg["content"])
    q = st.chat_input("Ask about this shortlist…", max_chars=ai.MAX_INPUT_CHARS)
    if q:
        q = q.strip()
        ss.chat.append({"role": "user", "content": q})
        if payload is None:
            reply = "There's no shortlist yet. Set a requirement in tab ① first."
        elif ai.looks_like_injection(q):
            reply = ("I can't change my instructions or reveal them. I can only help with this vendor shortlist. "
                     "Try: *Why is the top vendor ranked first?*")
        elif not AI_ON:
            t = ranked.iloc[0]
            reply = (f"AI chat is off ({status['msg']}). From the scoring: **{t.vendor_name}** ranks first with "
                     f"{t.score} points at ₹{t.unit_price_inr:,.2f}/{t.unit}. See tab ② for full details.")
        else:
            try:
                with st.spinner("Thinking…"):
                    reply = get_client(API_KEY, MODELS).answer(q, payload, ss.chat[:-1])
                ss.ai_calls += 1
                note = sc.cheaper_note(ranked, excluded)
                if note and re.search(r"cheap|lowest (?:price|cost)|\bL1\b", reply, re.I):
                    reply += f"\n\nℹ️ *Checked by the app: {note}*"
                bad = ai.unverified_figures(reply, payload)
                if bad:
                    reply += f"\n\n⚠️ *Check these figures, which aren't in the vendor data: {', '.join(bad)}*"
            except ai.AIError as e:
                reply = f"Sorry, the AI is unavailable right now ({e.message}). The ranking in tab ② still works."
        ss.chat.append({"role": "assistant", "content": reply})
        st.rerun()
    if ss.chat and st.button("Clear chat"):
        ss.chat = []
        st.rerun()

# ------------------------------------------------------------------ database tab
with t5:
    st.caption(ss.data_label)
    cat_f = st.multiselect("Filter categories", categories, default=categories)
    view = df[df.category.isin(cat_f)]
    st.dataframe(view, hide_index=True, use_container_width=True, height=460)
    st.caption(f"{len(view)} vendors · {view.category.nunique()} categories")

calls_slot.caption(f"AI calls this session: {ss.ai_calls}/{ai.MAX_CALLS_PER_SESSION}")
