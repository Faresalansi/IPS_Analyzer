from flask import Flask, request, jsonify, send_from_directory
import pickle, numpy as np, os

app = Flask(__name__, static_folder=".")

# ── تحميل الموديل والأدوات ──────────────────────────────────
BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, "model.pkl"),          "rb") as f: model          = pickle.load(f)
with open(os.path.join(BASE, "le_label.pkl"),        "rb") as f: le_label       = pickle.load(f)
with open(os.path.join(BASE, "features.pkl"),        "rb") as f: features       = pickle.load(f)
with open(os.path.join(BASE, "short_threshold.pkl"), "rb") as f: short_threshold= pickle.load(f)
with open(os.path.join(BASE, "attack_severity.pkl"), "rb") as f: attack_severity= pickle.load(f)

# ── thresholds metadata للـ frontend ───────────────────────
FEATURE_META = {
    "Fwd IAT Mean": {
        "label": "Forward IAT Mean",
        "unit": "µs",
        "description": "متوسط الزمن بين الحزم الأمامية",
        "thresholds": [
            {"op": "<=", "value": 0,      "level": "danger",  "text": "قيمة صفرية — نمط هجوم محتمل"},
            {"op": "<",  "value": 1000,   "level": "warning", "text": "IAT منخفض جداً — تدفق مكثف مريب"},
            {"op": "<",  "value": 100000, "level": "ok",      "text": "IAT طبيعي — حركة مرور اعتيادية"},
            {"op": ">=", "value": 100000, "level": "safe",    "text": "IAT مرتفع — حركة هادئة جداً"},
        ]
    },
    "Bwd IAT Mean": {
        "label": "Backward IAT Mean",
        "unit": "µs",
        "description": "متوسط الزمن بين الحزم العكسية",
        "thresholds": [
            {"op": "<=", "value": 0,      "level": "danger",  "text": "لا ردود عكسية — مشبوه"},
            {"op": "<",  "value": 1000,   "level": "warning", "text": "استجابة عكسية سريعة جداً"},
            {"op": "<",  "value": 100000, "level": "ok",      "text": "استجابة طبيعية"},
            {"op": ">=", "value": 100000, "level": "safe",    "text": "استجابة بطيئة — طبيعي"},
        ]
    },
    "Average Packet Size": {
        "label": "Average Packet Size",
        "unit": "bytes",
        "description": "متوسط حجم الحزمة",
        "thresholds": [
            {"op": "<",  "value": 50,   "level": "danger",  "text": "حزم صغيرة جداً — نمط SYN Flood أو PortScan"},
            {"op": "<",  "value": 200,  "level": "warning", "text": "حزم صغيرة — راقب النشاط"},
            {"op": "<",  "value": 1000, "level": "ok",      "text": "حجم حزمة طبيعي"},
            {"op": ">=", "value": 1000, "level": "safe",    "text": "حزم كبيرة — نقل بيانات اعتيادي"},
        ]
    },
    "SynOnlyRatio": {
        "label": "SYN-Only Ratio",
        "unit": "",
        "description": "نسبة الـ flows التي تحتوي SYN فقط بدون ACK",
        "thresholds": [
            {"op": ">=", "value": 0.9, "level": "danger",  "text": "SYN Flood مرجح — حجب خدمة"},
            {"op": ">=", "value": 0.5, "level": "warning", "text": "نسبة SYN مرتفعة — مشبوه"},
            {"op": ">=", "value": 0.1, "level": "ok",      "text": "بعض حزم SYN — طبيعي نسبياً"},
            {"op": "<",  "value": 0.1, "level": "safe",    "text": "SYN نادر — حركة طبيعية"},
        ]
    },
    "Flow Duration": {
        "label": "Flow Duration",
        "unit": "µs",
        "description": "مدة الـ flow الكاملة",
        "thresholds": [
            {"op": "<",  "value": short_threshold,       "level": "danger",  "text": f"أقل من threshold ({short_threshold:.0f}µs) — short flow مشبوه"},
            {"op": "<",  "value": short_threshold * 10,  "level": "warning", "text": "flow قصير نسبياً"},
            {"op": "<",  "value": short_threshold * 100, "level": "ok",      "text": "flow متوسط — طبيعي"},
            {"op": ">=", "value": short_threshold * 100, "level": "safe",    "text": "flow طويل — اتصال مستمر اعتيادي"},
        ]
    },
    "Flow IAT Mean": {
        "label": "Flow IAT Mean",
        "unit": "µs",
        "description": "متوسط الزمن بين الحزم في الـ flow",
        "thresholds": [
            {"op": "<=", "value": 0,      "level": "danger",  "text": "IAT صفري — حزم متتالية فورية"},
            {"op": "<",  "value": 1000,   "level": "warning", "text": "IAT منخفض — تدفق مكثف"},
            {"op": "<",  "value": 100000, "level": "ok",      "text": "IAT طبيعي"},
            {"op": ">=", "value": 100000, "level": "safe",    "text": "IAT مرتفع — حركة هادئة"},
        ]
    },
    "Flow IAT Std": {
        "label": "Flow IAT Std",
        "unit": "µs",
        "description": "انحراف معياري للـ IAT — يقيس التذبذب",
        "thresholds": [
            {"op": "<",  "value": 100,    "level": "danger",  "text": "تذبذب منعدم — حركة آلية ثابتة"},
            {"op": "<",  "value": 10000,  "level": "warning", "text": "تذبذب منخفض — محتمل آلي"},
            {"op": "<",  "value": 500000, "level": "ok",      "text": "تذبذب طبيعي"},
            {"op": ">=", "value": 500000, "level": "safe",    "text": "تذبذب عالٍ — حركة بشرية متنوعة"},
        ]
    },
    "Flow Packets/s": {
        "label": "Flow Packets/s",
        "unit": "pkt/s",
        "description": "عدد الحزم في الثانية — يُستخدم لحساب TrafficIntensity",
        "thresholds": [
            {"op": ">=", "value": 1000, "level": "danger",  "text": "≥1000 pkt/s — TrafficIntensity=2 (عالٍ جداً، DDoS محتمل)"},
            {"op": ">=", "value": 100,  "level": "warning", "text": "≥100 pkt/s — TrafficIntensity=1 (متوسط، راقب)"},
            {"op": "<",  "value": 100,  "level": "safe",    "text": "<100 pkt/s — TrafficIntensity=0 (طبيعي)"},
        ]
    },
    "Flow Bytes/s": {
        "label": "Flow Bytes/s",
        "unit": "B/s",
        "description": "عدد البايتات في الثانية — يُستخدم لحساب TrafficIntensity",
        "thresholds": [
            {"op": ">=", "value": 100000, "level": "danger",  "text": "≥100,000 B/s — TrafficIntensity=2 (عالٍ جداً)"},
            {"op": ">=", "value": 10000,  "level": "warning", "text": "≥10,000 B/s — TrafficIntensity=1 (متوسط)"},
            {"op": "<",  "value": 10000,  "level": "safe",    "text": "<10,000 B/s — TrafficIntensity=0 (طبيعي)"},
        ]
    },
    "TrafficIntensity": {
        "label": "Traffic Intensity (محسوب)",
        "unit": "0–2",
        "description": "يُحسب تلقائياً: max(PacketLevel, ByteLevel)",
        "thresholds": [
            {"op": ">=", "value": 2, "level": "danger",  "text": "كثافة عالية جداً — DDoS محتمل"},
            {"op": ">=", "value": 1, "level": "warning", "text": "كثافة متوسطة — راقب"},
            {"op": "<",  "value": 1, "level": "safe",    "text": "كثافة منخفضة — طبيعي"},
        ]
    },
    "Burstiness": {
        "label": "Burstiness",
        "unit": "0 or 1",
        "description": "هل الحركة متقطعة؟ (Std/Mean > 0.5)",
        "thresholds": [
            {"op": "==", "value": 1, "level": "warning", "text": "حركة متقطعة — نمط burst مشبوه"},
            {"op": "==", "value": 0, "level": "safe",    "text": "حركة منتظمة — طبيعي"},
        ]
    },
    "BackwardZeroLevel": {
        "label": "Backward Zero Level",
        "unit": "0 or 1",
        "description": "هل 90%+ من الـ flows ليس لها ردود عكسية؟",
        "thresholds": [
            {"op": "==", "value": 1, "level": "danger",  "text": "لا ردود — one-way traffic مشبوه جداً"},
            {"op": "==", "value": 0, "level": "safe",    "text": "ردود موجودة — اتصال ثنائي الاتجاه"},
        ]
    },
    "LowPacketLevel": {
        "label": "Low Packet Level",
        "unit": "0 or 1",
        "description": "هل 90%+ من الـ flows تحتوي ≤3 حزم؟",
        "thresholds": [
            {"op": "==", "value": 1, "level": "danger",  "text": "flows بحزم قليلة جداً — PortScan أو SYN Flood"},
            {"op": "==", "value": 0, "level": "safe",    "text": "flows بحزم كافية — طبيعي"},
        ]
    },
    "ShortFlowLevel": {
        "label": "Short Flow Level",
        "unit": "0 or 1",
        "description": f"هل 90%+ من الـ flows أقصر من threshold ({short_threshold:.0f}µs)؟",
        "thresholds": [
            {"op": "==", "value": 1, "level": "danger",  "text": "flows قصيرة جداً — نمط هجوم واضح"},
            {"op": "==", "value": 0, "level": "safe",    "text": "flows طبيعية الطول"},
        ]
    },
    "PortDiversityLevel": {
        "label": "Port Diversity Level",
        "unit": "0 or 1",
        "description": "هل عدد المنافذ المستهدفة (DistinctDestinationPort) ≥ 100؟",
        "thresholds": [
            {"op": "==", "value": 1, "level": "danger",  "text": "≥100 منفذ مستهدف — نمط PortScan واضح"},
            {"op": "==", "value": 0, "level": "safe",    "text": "منافذ محدودة — حركة طبيعية"},
        ]
    },
}


def compute_derived(data: dict) -> dict:
    """يحسب TrafficIntensity و Burstiness و Levels من القيم الخام."""
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

    # Levels (binary flags — ratio-based)
    d["BackwardZeroLevel"]  = int(float(d.get("BackwardZeroLevel",  0)) >= 0.9)
    d["LowPacketLevel"]     = int(float(d.get("LowPacketLevel",     0)) >= 0.9)
    d["ShortFlowLevel"]     = int(float(d.get("ShortFlowLevel",     0)) >= 0.9)

    # PortDiversityLevel — يعتمد على DistinctDestinationPort >= 100
    d["PortDiversityLevel"] = int(float(d.get("DistinctDestinationPort", 0)) >= 100)

    return d


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/meta")
def meta():
    """يرجع metadata الـ features والـ threshold."""
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

    # حساب الـ derived features
    enriched = compute_derived(raw)

    # بناء vector
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

    # تفسير كل feature
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