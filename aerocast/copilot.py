"""Module 7 - Explanation Copilot.

Turns VERIFIED model outputs into (a) a confidence tier, (b) top weather drivers,
and (c) a plain-English ATC brief. The language model only describes numbers the
deterministic pipeline already produced - it never computes or decides. If
ANTHROPIC_API_KEY is set and the SDK is installed, it polishes the brief; offline
it uses a deterministic template so the demo never breaks.

PortSync ancestor: Operations Copilot + Human approval.
"""
import os

_PRETTY = {
    "crosswind": "crosswind", "crosswind_gust": "gust crosswind",
    "tailwind": "tailwind", "headwind": "headwind", "wspd": "wind speed",
    "wpgt": "wind gust", "prcp": "precipitation", "snow": "snow",
    "pres": "pressure", "temp": "temperature", "coco": "weather condition",
    "aar_roll3_lag": "recent arrival throughput",
}


def _pretty(feature: str) -> str:
    base = feature.split("_lag")[0]
    lead = feature.split("_lag")[1] if "_lag" in feature else None
    name = _PRETTY.get(base, base)
    return f"{name} ({lead}h prior)" if lead else name


def top_drivers(importances: dict, k: int = 3) -> list[str]:
    return [_pretty(f) for f, v in list(importances.items())[:k] if v > 0][:k]


def confidence_tier(interval_width: float, typical: float) -> str:
    r = interval_width / max(typical, 1e-6)
    if r < 0.5:
        return "High"
    if r < 1.0:
        return "Medium"
    return "Low"


def template_brief(airport, dt, aar_pred, lo, hi, typical, drivers, n_risk, conf) -> str:
    rel = "below" if aar_pred < typical - 0.5 else ("above" if aar_pred > typical + 0.5 else "around")
    s = (f"{airport} @ {dt:%Y-%m-%d %H:%M}: forecast arrival capacity "
         f"{aar_pred:.1f}/h (range {lo:.1f}-{hi:.1f}), {rel} the typical {typical:.1f}/h. ")
    if drivers:
        s += "Main drivers: " + ", ".join(drivers) + ". "
    s += f"{int(n_risk)} arriving flight(s) flagged severe-delay risk. "
    s += f"Confidence {conf}. "
    s += "Low confidence - controller review advised." if conf == "Low" else "Controller decides whether to pre-emptively trim the arrival rate."
    return s


def brief(airport, dt, aar_pred, lo, hi, typical, drivers, n_risk, conf,
          use_llm: bool = True) -> str:
    base = template_brief(airport, dt, aar_pred, lo, hi, typical, drivers, n_risk, conf)
    if not use_llm or not os.getenv("ANTHROPIC_API_KEY"):
        return base
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=160,
            system=("You are an air-traffic decision-support copilot. Rewrite the "
                    "structured brief into one crisp sentence for a controller. "
                    "Do not invent numbers; the human decides."),
            messages=[{"role": "user", "content": base}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return base
