from flask import Flask, request, jsonify, send_from_directory
import pickle, numpy as np, os

app = Flask(__name__, static_folder=".")

# ── Load model and tools ──────────────────────────────────
BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, "model.pkl"),          "rb") as f: model          = pickle.load(f)
with open(os.path.join(BASE, "le_label.pkl"),        "rb") as f: le_label       = pickle.load(f)
with open(os.path.join(BASE, "features.pkl"),        "rb") as f: features       = pickle.load(f)
with open(os.path.join(BASE, "short_threshold.pkl"), "rb") as f: short_threshold= pickle.load(f)
with open(os.path.join(BASE, "attack_severity.pkl"), "rb") as f: attack_severity= pickle.load(f)

# ── Feature threshold metadata for the frontend ───────────────────────
FEATURE_META = {
    "Fwd IAT Mean": {
        "label": "Forward IAT Mean",
        "unit": "µs",
        "description": "Mean time between forward packets",
        "thresholds": [
            {"op": "<=", "value": 0,      "level": "danger",  "text": "Zero value — possible attack pattern"},
            {"op": "<",  "value": 1000,   "level": "warning", "text": "Very low IAT — suspiciously dense flow"},
            {"op": "<",  "value": 100000, "level": "ok",      "text": "Normal IAT — regular traffic"},
            {"op": ">=", "value": 100000, "level": "safe",    "text": "High IAT — very calm traffic"},
        ]
    },
    "Bwd IAT Mean": {
        "label": "Backward IAT Mean",
        "unit": "µs",
        "description": "Mean time between backward packets",
        "thresholds": [
            {"op": "<=", "value": 0,      "level": "danger",  "text": "No backward responses — suspicious"},
            {"op": "<",  "value": 1000,   "level": "warning", "text": "Very fast backward response"},
            {"op": "<",  "value": 100000, "level": "ok",      "text": "Normal response time"},
            {"op": ">=", "value": 100000, "level": "safe",    "text": "Slow response — normal"},
        ]
    },
    "Average Packet Size": {
        "label": "Average Packet Size",
        "unit": "bytes",
        "description": "Mean packet size",
        "thresholds": [
            {"op": "<",  "value": 50,   "level": "danger",  "text": "Very small packets — SYN Flood or PortScan pattern"},
            {"op": "<",  "value": 200,  "level": "warning", "text": "Small packets — monitor activity"},
            {"op": "<",  "value": 1000, "level": "ok",      "text": "Normal packet size"},
            {"op": ">=", "value": 1000, "level": "safe",    "text": "Large packets — normal data transfer"},
        ]
    },
    "SynOnlyRatio": {
        "label": "SYN-Only Ratio",
        "unit": "",
        "description": "Ratio of flows containing SYN only without ACK",
        "thresholds": [
            {"op": ">=", "value": 0.9, "level": "danger",  "text": "SYN Flood likely — denial of service"},
            {"op": ">=", "value": 0.5, "level": "warning", "text": "High SYN ratio — suspicious"},
            {"op": ">=", "value": 0.1, "level": "ok",      "text": "Some SYN packets — relatively normal"},
            {"op": "<",  "value": 0.1, "level": "safe",    "text": "Rare SYN — normal traffic"},
        ]
    },
    "Flow Duration": {
        "label": "Flow Duration",
        "unit": "µs",
        "description": "Total duration of the flow",
        "thresholds": [
            {"op": "<",  "value": short_threshold,       "level": "danger",  "text": f"Below threshold ({short_threshold:.0f}µs) — suspicious short flow"},
            {"op": "<",  "value": short_threshold * 10,  "level": "warning", "text": "Relatively short flow"},
            {"op": "<",  "value": short_threshold * 100, "level": "ok",      "text": "Medium flow — normal"},
            {"op": ">=", "value": short_threshold * 100, "level": "safe",    "text": "Long flow — sustained normal connection"},
        ]
    },
    "Flow IAT Mean": {
        "label": "Flow IAT Mean",
        "unit": "µs",
        "description": "Mean inter-arrival time between packets in the flow",
        "thresholds": [
            {"op": "<=", "value": 0,      "level": "danger",  "text": "Zero IAT — back-to-back instant packets"},
            {"op": "<",  "value": 1000,   "level": "warning", "text": "Low IAT — dense flow"},
            {"op": "<",  "value": 100000, "level": "ok",      "text": "Normal IAT"},
            {"op": ">=", "value": 100000, "level": "safe",    "text": "High IAT — calm traffic"},
        ]
    },
    "Flow IAT Std": {
        "label": "Flow IAT Std",
        "unit": "µs",
        "description": "Standard deviation of IAT — measures variability",
        "thresholds": [
            {"op": "<",  "value": 100,    "level": "danger",  "text": "No variance — fixed automated traffic"},
            {"op": "<",  "value": 10000,  "level": "warning", "text": "Low variance — possibly automated"},
            {"op": "<",  "value": 500000, "level": "ok",      "text": "Normal variance"},
            {"op": ">=", "value": 500000, "level": "safe",    "text": "High variance — diverse human traffic"},
        ]
    },
    "Flow Packets/s": {
        "label": "Flow Packets/s",
        "unit": "pkt/s",
        "description": "Packets per second — used to compute TrafficIntensity",
        "thresholds": [
            {"op": ">=", "value": 1000, "level": "danger",  "text": "≥1000 pkt/s — TrafficIntensity=2 (very high, possible DDoS)"},
            {"op": ">=", "value": 100,  "level": "warning", "text": "≥100 pkt/s — TrafficIntensity=1 (medium, monitor)"},
            {"op": "<",  "value": 100,  "level": "safe",    "text": "<100 pkt/s — TrafficIntensity=0 (normal)"},
        ]
    },
    "Flow Bytes/s": {
        "label": "Flow Bytes/s",
        "unit": "B/s",
        "description": "Bytes per second — used to compute TrafficIntensity",
        "thresholds": [
            {"op": ">=", "value": 100000, "level": "danger",  "text": "≥100,000 B/s — TrafficIntensity=2 (very high)"},
            {"op": ">=", "value": 10000,  "level": "warning", "text": "≥10,000 B/s — TrafficIntensity=1 (medium)"},
            {"op": "<",  "value": 10000,  "level": "safe",    "text": "<10,000 B/s — TrafficIntensity=0 (normal)"},
        ]
    },
    "TrafficIntensity": {
        "label": "Traffic Intensity (computed)",
        "unit": "0–2",
        "description": "Auto-computed: max(PacketLevel, ByteLevel)",
        "thresholds": [
            {"op": ">=", "value": 2, "level": "danger",  "text": "Very high intensity — possible DDoS"},
            {"op": ">=", "value": 1, "level": "warning", "text": "Medium intensity — monitor"},
            {"op": "<",  "value": 1, "level": "safe",    "text": "Low intensity — normal"},
        ]
    },
    "Burstiness": {
        "label": "Burstiness",
        "unit": "0 or 1",
        "description": "Is the traffic bursty? (Std/Mean > 0.5)",
        "thresholds": [
            {"op": "==", "value": 1, "level": "warning", "text": "Bursty traffic — suspicious burst pattern"},
            {"op": "==", "value": 0, "level": "safe",    "text": "Steady traffic — normal"},
        ]
    },
    "BackwardZeroLevel": {
        "label": "Backward Zero Level",
        "unit": "0 or 1",
        "description": "Do 90%+ of flows have no backward responses?",
        "thresholds": [
            {"op": "==", "value": 1, "level": "danger",  "text": "No responses — very suspicious one-way traffic"},
            {"op": "==", "value": 0, "level": "safe",    "text": "Responses present — normal two-way connection"},
        ]
    },
    "LowPacketLevel": {
        "label": "Low Packet Level",
        "unit": "0 or 1",
        "description": "Do 90%+ of flows contain ≤3 packets?",
        "thresholds": [
            {"op": "==", "value": 1, "level": "danger",  "text": "Flows with very few packets — PortScan or SYN Flood"},
            {"op": "==", "value": 0, "level": "safe",    "text": "Sufficient packets in flows — normal"},
        ]
    },
    "ShortFlowLevel": {
        "label": "Short Flow Level",
        "unit": "0 or 1",
        "description": f"Do 90%+ of flows fall below the threshold ({short_threshold:.0f}µs)?",
        "thresholds": [
            {"op": "==", "value": 1, "level": "danger",  "text": "Very short flows — clear attack pattern"},
            {"op": "==", "value": 0, "level": "safe",    "text": "Normal flow length"},
        ]
    },
    "PortDiversityLevel": {
        "label": "Port Diversity Level",
        "unit": "0 or 1",
        "description": "Is the number of distinct destination ports ≥ 100?",
        "thresholds": [
            {"op": "==", "value": 1, "level": "danger",  "text": "≥100 ports targeted — clear PortScan pattern"},
            {"op": "==", "value": 0, "level": "safe",    "text": "Limited ports — normal traffic"},
        ]
    },
}


def compute_derived(data: dict) -> dict:
    """Computes TrafficIntensity, Burstiness, and level flags from raw values."""
    d = dict(data)

    # TrafficIntensity
    fp  = float(d.get("Flow Packets/s", 0))
    fb  = float(d.get("Flow Bytes/s",   0))
    pl  = 0
    if fp >= 1000: pl = 2
    elif fp >= 100: pl = 1
    bl  = 0
    if fb >= 100000: bl = 2
    elif fb >= 10000: bl = 1
    d["TrafficIntensity"] = max(pl, bl)

    # Burstiness
    iat_std  = float(d.get("Flow IAT Std",  0))
    iat_mean = float(d.get("Flow IAT Mean", 0))
    ratio    = (iat_std / iat_mean) if iat_mean != 0 else 0
    d["Burstiness"] = int(ratio > 0.5)

    # Level flags (binary flags — ratio-based)
    d["BackwardZeroLevel"]  = int(float(d.get("BackwardZeroLevel",  0)) >= 0.9)
    d["LowPacketLevel"]     = int(float(d.get("LowPacketLevel",     0)) >= 0.9)
    d["ShortFlowLevel"]     = int(float(d.get("ShortFlowLevel",     0)) >= 0.9)

    # PortDiversityLevel — based on DistinctDestinationPort >= 100
    d["PortDiversityLevel"] = int(float(d.get("DistinctDestinationPort", 0)) >= 100)

    return d


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/meta")
def meta():
    """Returns feature metadata and thresholds."""
    return jsonify({
        "features": features,
        "short_threshold": float(short_threshold),
        "feature_meta": FEATURE_META,
        "classes": list(le_label.classes_),
    })


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    raw  = body.get("features", {})

    # Compute derived features
    enriched = compute_derived(raw)

    # Build feature vector
    vec = []
    for feat in features:
        val = float(enriched.get(feat, 0))
        val = 0 if (np.isnan(val) or np.isinf(val)) else val
        vec.append(val)

    X       = np.array([vec])
    pred    = model.predict(X)[0]
    proba   = model.predict_proba(X)[0]
    label   = le_label.inverse_transform([pred])[0]
    ips     = attack_severity.get(label, "RATE LIMIT")

    # Explain each feature
    explanations = {}
    for feat in features:
        val   = enriched.get(feat, 0)
        meta  = FEATURE_META.get(feat, {})
        level = "ok"
        text  = ""
        for thr in meta.get("thresholds", []):
            op, tv = thr["op"], thr["value"]
            match = (
                (op == ">="  and float(val) >= tv) or
                (op == "<="  and float(val) <= tv) or
                (op == ">"   and float(val) >  tv) or
                (op == "<"   and float(val) <  tv) or
                (op == "=="  and float(val) == tv)
            )
            if match:
                level = thr["level"]
                text  = thr["text"]
                break
        explanations[feat] = {"value": val, "level": level, "text": text}

    return jsonify({
        "label":  label,
        "ips":    ips,
        "proba":  {cls: float(p) for cls, p in zip(le_label.classes_, proba)},
        "explanations": explanations,
        "used_values":  enriched,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
