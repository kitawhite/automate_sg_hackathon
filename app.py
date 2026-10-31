"""Module 8 - ATC decision-support dashboard (Streamlit).

Reads the artifacts written by `python -m aerocast.pipeline` and presents the
controller view: AAR forecast with confidence band, capacity-reduction alerts,
ranked severe-delay-risk flights, weather drivers, a plain-English brief, and an
honest metrics panel. Nothing is computed here - it only displays verified output.

Run:  streamlit run app.py   (after the pipeline has produced ./artifacts)

PortSync ancestor: the Streamlit operations dashboard.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

from aerocast import copilot, config

ART = Path(__file__).resolve().parent / "artifacts"

st.set_page_config(page_title="AeroCast AI", page_icon="airplane", layout="wide")


@st.cache_data
def load():
    def js(name):
        p = ART / name
        return json.loads(p.read_text()) if p.exists() else {}

    def csv(name):
        p = ART / name
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)
        for c in ("dt", "sched_arr_dt"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df

    return js("metrics.json"), js("context.json"), js("drivers.json"), \
        csv("aar_predictions.csv"), csv("risk_predictions.csv")


metrics, context, drivers, aar, risk = load()

st.title("AeroCast AI")
st.caption("Predictive Airport Capacity & Delay-Risk Copilot - the controller decides; the model advises.")

if not metrics:
    st.warning("No artifacts found. Run `python -m aerocast.pipeline` first.")
    st.stop()

# ---- honest-disclosure banner ------------------------------------------------
if metrics.get("honest_note"):
    st.info("Honesty note: " + metrics["honest_note"])

# ---- top metrics -------------------------------------------------------------
a = metrics.get("aar", {})
r = metrics.get("risk", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("AAR RMSE (model)", a.get("rmse_model", "-"),
          delta=f"{a.get('skill_vs_best_baseline', 0):+.1%} vs baseline", delta_color="off",
          help="Matches climatology - expected on a 0.5% flight sample; the predictive "
               "signal lives in delay risk (see the note above).")
c2.metric("AAR climatology RMSE", a.get("rmse_climatology", "-"))
c3.metric("Risk PR-AUC", r.get("pr_auc", "-"), help=f"base rate {r.get('positive_rate','-')}")
c4.metric("Risk recall @ top-20% flag", r.get("recall", "-"),
          help=f"flag top 20% by score (thr {r.get('threshold','-')}); "
               f"lift {r.get('lift_vs_random','-')}x vs random, precision {r.get('precision','-')}")

_span = metrics.get("data_span", ["?", "?"])
st.caption(f"Wind direction: {metrics.get('wind_direction_mode','?')} | "
           f"data {_span[0]} to {_span[1]} | test from {metrics.get('test_start','?')} | "
           f"airports {', '.join(context.get('airports', []))}")

# ---- controls ----------------------------------------------------------------
airports = sorted(aar["airport"].unique()) if not aar.empty else config.AIRPORTS
left, right = st.columns([1, 3])
with left:
    ap = st.selectbox("Airport", airports)
    sub = aar[aar["airport"] == ap].sort_values("dt") if not aar.empty else pd.DataFrame()
    if not sub.empty:
        days = sorted(sub["dt"].dt.date.unique())
        day = st.select_slider("Day (hold-out)", options=days, value=days[len(days) // 2])
    else:
        day = None

# ---- AAR forecast chart ------------------------------------------------------
st.subheader(f"Arrival Acceptance Rate forecast - {ap}")
if not sub.empty and day is not None:
    center = pd.Timestamp(day)
    win = sub[(sub["dt"] >= center - pd.Timedelta(days=1)) &
              (sub["dt"] < center + pd.Timedelta(days=2))]
    if HAVE_PLOTLY and not win.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=win["dt"], y=win["hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=win["dt"], y=win["lo"], fill="tonexty", line=dict(width=0),
                                 name="80% interval", fillcolor="rgba(224,134,30,0.18)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=win["dt"], y=win["typical"], name="Typical (climatology)",
                                 line=dict(color="#8893a8", dash="dot")))
        fig.add_trace(go.Scatter(x=win["dt"], y=win["pred"], name="Forecast", line=dict(color="#E0861E", width=3)))
        fig.add_trace(go.Scatter(x=win["dt"], y=win["aar"], name="Actual", mode="markers",
                                 marker=dict(color="#1E2761", size=6)))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h"), yaxis_title="arrivals / hour")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(win.set_index("dt")[["pred", "typical", "aar"]])

    # capacity-reduction alerts (forecast materially below typical)
    alerts = win[win["pred"] < win["typical"] - 0.5]
    if not alerts.empty:
        st.error(f"Capacity-reduction signal at {len(alerts)} hour(s): forecast AAR below typical. "
                 f"Earliest {alerts['dt'].min():%Y-%m-%d %H:%M}.")
    else:
        st.success("No capacity-reduction signal in this window.")

# ---- risky flights + drivers + brief ----------------------------------------
colA, colB = st.columns([3, 2])
with colA:
    st.subheader("Flights flagged severe-delay risk (> 45 min)")
    if not risk.empty:
        rsub = risk[risk["airport"] == ap].sort_values("proba", ascending=False).head(12)
        show = rsub[["sched_arr_dt", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "proba"]] \
            .rename(columns={"OP_UNIQUE_CARRIER": "carrier", "OP_CARRIER_FL_NUM": "flight",
                             "ORIGIN": "from", "proba": "risk"})
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.write("No risk predictions available.")

with colB:
    st.subheader("Why - top drivers")
    st.caption("Capacity (AAR):")
    for d in (drivers.get("drivers", {}).get("aar", []) or [])[:4]:
        st.write("- " + d)
    st.caption("Delay risk:")
    for d in (drivers.get("drivers", {}).get("risk", []) or [])[:4]:
        st.write("- " + d)
    st.subheader("Controller brief")
    if not sub.empty and day is not None:
        row = sub[sub["dt"].dt.date == pd.Timestamp(day).date()]
        if not row.empty:
            row = row.iloc[len(row) // 2]
            nrisk = 0
            if not risk.empty:
                day_risk = risk[(risk["airport"] == ap) &
                                (risk["dt"].dt.date == pd.Timestamp(day).date())]
                nrisk = int((day_risk["proba"] >= r.get("threshold", 0.5)).sum())
            txt = copilot.brief(ap, pd.Timestamp(row["dt"]), float(row["pred"]),
                                float(row["lo"]), float(row["hi"]), float(row["typical"]),
                                drivers.get("drivers", {}).get("aar", []), nrisk,
                                copilot.confidence_tier(float(row["hi"] - row["lo"]), max(float(row["typical"]), 1)),
                                use_llm=True)   # live Claude brief when ANTHROPIC_API_KEY is set (.env), else template
            st.write(txt)

# ---- failure case + honest panel --------------------------------------------
with st.expander("Honest evaluation - baselines, generalisation, calibration, failure case"):
    st.markdown("**AAR (regression)** - model vs persistence / climatology baselines:")
    st.json(metrics.get("aar"))
    st.markdown("**Delay risk (classification)** - the predictive centrepiece "
                "(Platt-calibrated; operating point = review the top 20% highest-risk arrivals):")
    st.json({k: v for k, v in metrics.get("risk", {}).items()
             if k not in ("per_airport", "leave_one_airport_out")})
    st.markdown("**Per-airport risk:**")
    st.write(metrics.get("risk", {}).get("per_airport", {}))
    st.markdown("**Leave-one-airport-out generalisation** "
                "(train on the *other* airport only, test on this one - guards the 'overfit to one airport' trap):")
    st.write(metrics.get("risk", {}).get("leave_one_airport_out", {}))
    st.markdown("**Worst AAR error (failure case):**")
    st.write(metrics.get("failure_case", {}))

st.caption("AeroCast AI - Challenge #5 scaffold. The language model only explains verified numbers; "
           "a qualified controller makes the final call.")
