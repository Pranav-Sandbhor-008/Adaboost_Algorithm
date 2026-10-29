import os
import pickle
import warnings

import numpy as np
from flask import Flask, Response, jsonify, request

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "adaboost.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Exact column order the model was trained with
FEATURES = [
    "age", "gender", "state", "family_income", "high_school_gpa", "sat_score",
    "act_score", "attendance_rate", "ap_courses", "extracurricular_count",
    "volunteer_hours", "leadership_positions", "coding_projects",
    "social_media_hours", "online_certifications", "essay_score",
    "recommendation_score", "interview_score",
]

# gender & state had zero importance in the trained model, so they are sent
# as a constant (0). Change here if you want to send real encoded values.
CONSTANTS = {"gender": 0, "state": 0}

# UI fields: key, label, group, min, max, step, default, unit, hint
FIELDS = [
    # Academics
    {"key": "high_school_gpa", "label": "High school GPA", "group": "Academics", "min": 0, "max": 4, "step": 0.01, "default": 3.4, "unit": "/ 4.0", "hint": "Unweighted GPA on a 4.0 scale"},
    {"key": "sat_score", "label": "SAT score", "group": "Academics", "min": 400, "max": 1600, "step": 10, "default": 1200, "unit": "/ 1600", "hint": "Total SAT score"},
    {"key": "act_score", "label": "ACT score", "group": "Academics", "min": 1, "max": 36, "step": 1, "default": 26, "unit": "/ 36", "hint": "Composite ACT score"},
    {"key": "ap_courses", "label": "AP courses taken", "group": "Academics", "min": 0, "max": 15, "step": 1, "default": 3, "unit": "courses", "hint": "Advanced Placement classes"},
    {"key": "attendance_rate", "label": "Attendance rate", "group": "Academics", "min": 0, "max": 100, "step": 1, "default": 93, "unit": "%", "hint": "Percentage of school days attended"},
    # Activities
    {"key": "extracurricular_count", "label": "Extracurricular activities", "group": "Activities & Skills", "min": 0, "max": 15, "step": 1, "default": 3, "unit": "activities", "hint": "Clubs, sports, music, etc."},
    {"key": "volunteer_hours", "label": "Volunteer hours", "group": "Activities & Skills", "min": 0, "max": 500, "step": 5, "default": 60, "unit": "hours", "hint": "Total community service hours"},
    {"key": "leadership_positions", "label": "Leadership positions", "group": "Activities & Skills", "min": 0, "max": 10, "step": 1, "default": 1, "unit": "roles", "hint": "Captain, president, lead, etc."},
    {"key": "coding_projects", "label": "Coding projects", "group": "Activities & Skills", "min": 0, "max": 20, "step": 1, "default": 2, "unit": "projects", "hint": "Personal or school projects"},
    {"key": "online_certifications", "label": "Online certifications", "group": "Activities & Skills", "min": 0, "max": 20, "step": 1, "default": 2, "unit": "certs", "hint": "Coursera, edX, Google, etc."},
    # Evaluations
    {"key": "essay_score", "label": "Essay score", "group": "Evaluations", "min": 0, "max": 100, "step": 1, "default": 70, "unit": "/ 100", "hint": "Quality of the application essay"},
    {"key": "recommendation_score", "label": "Recommendation score", "group": "Evaluations", "min": 0, "max": 100, "step": 1, "default": 70, "unit": "/ 100", "hint": "Strength of recommendation letters"},
    {"key": "interview_score", "label": "Interview score", "group": "Evaluations", "min": 0, "max": 100, "step": 1, "default": 70, "unit": "/ 100", "hint": "Performance in the interview"},
    # Personal
    {"key": "age", "label": "Age", "group": "Personal (optional)", "min": 14, "max": 30, "step": 1, "default": 18, "unit": "years", "hint": "Applicant age"},
    {"key": "family_income", "label": "Family income", "group": "Personal (optional)", "min": 0, "max": 500000, "step": 1000, "default": 60000, "unit": "per year", "hint": "Annual household income"},
    {"key": "social_media_hours", "label": "Social media time", "group": "Personal (optional)", "min": 0, "max": 12, "step": 0.5, "default": 2, "unit": "hrs / day", "hint": "Average daily usage"},
]
FIELD_MAP = {f["key"]: f for f in FIELDS}

# Benchmarks used to explain results in plain language
BENCHMARKS = {
    "high_school_gpa": ("GPA", 3.5, "a GPA of {v:.2f} is solid"),
    "sat_score": ("SAT score", 1250, "an SAT score of {v:.0f} is competitive"),
    "ap_courses": ("AP coursework", 3, "{v:.0f} AP courses show academic rigor"),
    "extracurricular_count": ("Extracurriculars", 4, "{v:.0f} extracurricular activities show a well-rounded profile"),
    "volunteer_hours": ("Volunteering", 90, "{v:.0f} volunteer hours show community commitment"),
    "leadership_positions": ("Leadership", 1, "holding {v:.0f} leadership role(s) stands out"),
    "coding_projects": ("Coding projects", 3, "{v:.0f} coding projects show practical skills"),
    "online_certifications": ("Certifications", 3, "{v:.0f} certifications show initiative"),
    "essay_score": ("Essay", 75, "an essay score of {v:.0f} is strong"),
    "recommendation_score": ("Recommendations", 75, "recommendation score of {v:.0f} is strong"),
    "interview_score": ("Interview", 75, "an interview score of {v:.0f} is impressive"),
}

# What-if steps used for "how to improve" suggestions
IMPROVE_STEPS = {
    "high_school_gpa": (0.3, "Raise your GPA by about 0.3"),
    "sat_score": (100, "Improve your SAT score by around 100 points"),
    "ap_courses": (2, "Take 2 more AP courses"),
    "extracurricular_count": (2, "Join 2 more extracurricular activities"),
    "volunteer_hours": (30, "Add roughly 30 more volunteer hours"),
    "leadership_positions": (1, "Take on 1 more leadership role"),
    "coding_projects": (2, "Build 2 more coding projects"),
    "online_certifications": (2, "Earn 2 more online certifications"),
    "essay_score": (10, "Strengthen your essay by about 10 points"),
    "recommendation_score": (10, "Get stronger recommendation letters (+10)"),
    "interview_score": (10, "Improve your interview performance by about 10 points"),
}


def clean(payload):
    """Validate + clamp incoming values to the allowed ranges."""
    values = {}
    for f in FIELDS:
        raw = payload.get(f["key"], f["default"])
        try:
            v = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"'{f['label']}' must be a number.")
        if np.isnan(v):
            raise ValueError(f"'{f['label']}' must be a number.")
        values[f["key"]] = min(max(v, f["min"]), f["max"])
    return values


def proba(values):
    row = [CONSTANTS.get(k, values.get(k, 0)) for k in FEATURES]
    return float(model.predict_proba(np.array([row], dtype=float))[0][1])


def build_report(values):
    p = proba(values)
    pct = round(p * 100, 1)
    admitted = p >= 0.5

    if p >= 0.80:
        tier, headline = "excellent", "Excellent chances"
        summary = "Your profile is very strong. The model is highly confident this application would be accepted."
    elif p >= 0.60:
        tier, headline = "good", "Good chances"
        summary = "Your profile looks promising. The model leans towards acceptance, though a few areas could still be strengthened."
    elif p >= 0.40:
        tier, headline = "borderline", "Borderline, could go either way"
        summary = "Your profile sits right on the edge. Small improvements in the key areas below could tip the result in your favour."
    elif p >= 0.20:
        tier, headline = "low", "Low chances right now"
        summary = "The model currently leans towards rejection, but there is clear room to improve your profile."
    else:
        tier, headline = "veryLow", "Very low chances right now"
        summary = "The model sees a big gap between this profile and typical accepted applicants. Focus on the biggest levers below."

    strengths, gaps = [], []
    for key, (name, bench, text) in BENCHMARKS.items():
        v = values[key]
        if v >= bench:
            strengths.append({"name": name, "text": text.format(v=v)})
        else:
            gaps.append({"name": name, "text": f"{name} ({v:g}) is below the typical level of about {bench:g}"})

    tips = []
    for key, (step, text) in IMPROVE_STEPS.items():
        f = FIELD_MAP[key]
        new_v = min(values[key] + step, f["max"])
        if new_v == values[key]:
            continue
        alt = dict(values)
        alt[key] = new_v
        gain = (proba(alt) - p) * 100
        if gain > 0.4:
            tips.append({"text": text, "gain": round(gain, 1)})
    tips.sort(key=lambda t: t["gain"], reverse=True)

    return {
        "probability": pct,
        "admitted": admitted,
        "tier": tier,
        "headline": headline,
        "summary": summary,
        "verdict": "Likely to be ACCEPTED" if admitted else "Likely to be REJECTED",
        "strengths": strengths[:6],
        "gaps": gaps[:6],
        "tips": tips[:3],
    }


app = Flask(__name__)


@app.route("/")
def home():
    import json
    return Response(PAGE.replace("__FIELDS__", json.dumps(FIELDS)), mimetype="text/html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        return jsonify(build_report(clean(payload)))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Something went wrong while predicting. Please try again."}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


PAGE = r"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Admission Predictor</title>
<script>
  (function () {
    var t = localStorage.getItem("theme");
    if (!t) t = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", t);
  })();
</script>
<style>
  :root {
    --bg: #f4f6fb; --card: #ffffff; --text: #0f172a; --muted: #64748b; --border: #e2e8f0;
    --primary: #4f46e5; --primary-2: #7c3aed; --track: #e2e8f0; --shadow: 0 10px 30px rgba(15,23,42,.08);
    --good: #16a34a; --warn: #d97706; --bad: #dc2626; --chip-good: #dcfce7; --chip-bad: #fee2e2; --chip-tip: #e0e7ff;
    --input: #f8fafc;
  }
  [data-theme="dark"] {
    --bg: #0b1020; --card: #131a2e; --text: #e5e9f5; --muted: #94a3b8; --border: #243049;
    --primary: #818cf8; --primary-2: #a78bfa; --track: #25304a; --shadow: 0 10px 30px rgba(0,0,0,.45);
    --good: #4ade80; --warn: #fbbf24; --bad: #f87171; --chip-good: #12372a; --chip-bad: #3f1a1f; --chip-tip: #1f2650;
    --input: #0e1427;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0; font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); transition: background .3s, color .3s; line-height: 1.5;
  }
  body::before {
    content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background: radial-gradient(600px 300px at 10% -5%, color-mix(in srgb, var(--primary) 18%, transparent), transparent),
                radial-gradient(500px 300px at 95% 0%, color-mix(in srgb, var(--primary-2) 15%, transparent), transparent);
  }
  header {
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    max-width: 1200px; margin: 0 auto; padding: 20px 20px 8px;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .logo {
    width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center; font-size: 22px;
    background: linear-gradient(135deg, var(--primary), var(--primary-2)); color: #fff; box-shadow: var(--shadow);
  }
  .brand h1 { font-size: 20px; margin: 0; letter-spacing: -.2px; }
  .brand p { margin: 0; font-size: 13px; color: var(--muted); }
  .icon-btn {
    border: 1px solid var(--border); background: var(--card); color: var(--text); cursor: pointer;
    width: 42px; height: 42px; border-radius: 12px; font-size: 18px; transition: transform .15s, border-color .2s;
  }
  .icon-btn:hover { transform: translateY(-1px); border-color: var(--primary); }
  main { max-width: 1200px; margin: 0 auto; padding: 12px 20px 60px; display: grid; grid-template-columns: 1.35fr 1fr; gap: 22px; align-items: start; }
  @media (max-width: 960px) { main { grid-template-columns: 1fr; } }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 18px; padding: 22px; box-shadow: var(--shadow); transition: background .3s, border-color .3s; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 8px; }
  .toolbar .label { font-size: 13px; color: var(--muted); margin-right: 4px; }
  .pill {
    border: 1px solid var(--border); background: var(--input); color: var(--text); padding: 7px 13px; border-radius: 999px;
    cursor: pointer; font-size: 13px; font-weight: 500; transition: all .15s;
  }
  .pill:hover { border-color: var(--primary); color: var(--primary); transform: translateY(-1px); }
  .group { margin-top: 22px; }
  .group h3 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 0 0 12px; display: flex; align-items: center; gap: 8px; }
  .group h3::after { content: ""; flex: 1; height: 1px; background: var(--border); }
  .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px 20px; }
  @media (max-width: 620px) { .grid { grid-template-columns: 1fr; } }
  .field label { display: flex; justify-content: space-between; font-size: 14px; font-weight: 600; margin-bottom: 6px; }
  .field label span { font-weight: 400; color: var(--muted); font-size: 12px; }
  .row { display: flex; align-items: center; gap: 10px; }
  .num {
    width: 92px; padding: 8px 10px; border-radius: 10px; border: 1px solid var(--border); background: var(--input);
    color: var(--text); font-size: 14px; font-weight: 600; text-align: right; outline: none; transition: border-color .2s, box-shadow .2s;
  }
  .num:focus { border-color: var(--primary); box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary) 25%, transparent); }
  input[type=range] { flex: 1; -webkit-appearance: none; appearance: none; height: 6px; border-radius: 99px; outline: none;
    background: linear-gradient(90deg, var(--primary) var(--p, 50%), var(--track) var(--p, 50%)); }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 20px; height: 20px; border-radius: 50%; background: #fff; border: 3px solid var(--primary); cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,.25); transition: transform .15s; }
  input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.15); }
  input[type=range]::-moz-range-thumb { width: 14px; height: 14px; border-radius: 50%; background: #fff; border: 3px solid var(--primary); cursor: pointer; }
  .hint { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .actions { display: flex; gap: 12px; margin-top: 28px; flex-wrap: wrap; }
  .btn { border: 0; cursor: pointer; font-weight: 600; font-size: 15px; padding: 13px 22px; border-radius: 12px; transition: transform .15s, box-shadow .2s, opacity .2s; }
  .btn.primary { flex: 1; min-width: 180px; color: #fff; background: linear-gradient(135deg, var(--primary), var(--primary-2)); box-shadow: 0 8px 20px color-mix(in srgb, var(--primary) 40%, transparent); }
  .btn.primary:hover { transform: translateY(-2px); }
  .btn.primary:disabled { opacity: .7; cursor: wait; transform: none; }
  .btn.ghost { background: var(--input); color: var(--text); border: 1px solid var(--border); }
  .btn.ghost:hover { border-color: var(--primary); }
  aside { position: sticky; top: 16px; display: grid; gap: 22px; }
  @media (max-width: 960px) { aside { position: static; } }
  .empty { text-align: center; padding: 30px 10px; color: var(--muted); }
  .empty .big { font-size: 46px; margin-bottom: 8px; }
  .gauge-wrap { display: grid; place-items: center; position: relative; margin: 6px 0 4px; }
  .gauge { width: 190px; height: 190px; transform: rotate(-90deg); }
  .gauge circle { fill: none; stroke-width: 12; stroke-linecap: round; }
  .gauge .bg { stroke: var(--track); }
  .gauge .fg { stroke: var(--c, var(--primary)); stroke-dasharray: 339.29; stroke-dashoffset: 339.29; transition: stroke-dashoffset 1.1s cubic-bezier(.2,.8,.2,1), stroke .4s; }
  .gauge-text { position: absolute; text-align: center; }
  .gauge-text b { font-size: 40px; letter-spacing: -1px; color: var(--c, var(--text)); }
  .gauge-text small { display: block; color: var(--muted); font-size: 12px; margin-top: -4px; }
  .verdict { text-align: center; }
  .verdict .badge { display: inline-block; padding: 6px 14px; border-radius: 999px; font-weight: 700; font-size: 13px; color: #fff; background: var(--c, var(--primary)); }
  .verdict h2 { margin: 12px 0 6px; font-size: 22px; letter-spacing: -.3px; }
  .verdict p { margin: 0; color: var(--muted); font-size: 14px; }
  .sec { margin-top: 20px; }
  .sec h4 { margin: 0 0 8px; font-size: 14px; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; }
  .chip { padding: 6px 11px; border-radius: 10px; font-size: 12.5px; line-height: 1.35; }
  .chip.good { background: var(--chip-good); color: var(--good); }
  .chip.bad { background: var(--chip-bad); color: var(--bad); }
  .tip { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 11px 13px; border-radius: 12px; background: var(--chip-tip); margin-bottom: 8px; font-size: 13.5px; }
  .tip b { color: var(--primary); white-space: nowrap; }
  .share { display: flex; gap: 10px; margin-top: 18px; }
  .share .btn { flex: 1; padding: 10px; font-size: 13px; }
  .hist-title { display: flex; justify-content: space-between; align-items: center; margin: 0 0 10px; font-size: 15px; }
  .hist-title button { background: none; border: 0; color: var(--muted); cursor: pointer; font-size: 12px; }
  .hist { display: grid; gap: 8px; }
  .hist-item { display: flex; justify-content: space-between; align-items: center; padding: 10px 12px; border: 1px solid var(--border); border-radius: 12px; font-size: 13px; background: var(--input); }
  .hist-item b { color: var(--c); }
  .spinner { display: inline-block; width: 16px; height: 16px; border: 2.5px solid rgba(255,255,255,.4); border-top-color: #fff; border-radius: 50%; animation: spin .7s linear infinite; vertical-align: -3px; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .fade-in { animation: fade .5s ease both; }
  @keyframes fade { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
  #toast { position: fixed; left: 50%; bottom: 26px; transform: translateX(-50%) translateY(30px); background: var(--text); color: var(--bg); padding: 11px 18px; border-radius: 12px; font-size: 14px; opacity: 0; pointer-events: none; transition: all .3s; z-index: 10; }
  #toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
  .disclaimer { font-size: 12px; color: var(--muted); text-align: center; margin-top: 4px; }
  .tier-excellent { --c: var(--good); } .tier-good { --c: #22c55e; } .tier-borderline { --c: var(--warn); }
  .tier-low { --c: #f97316; } .tier-veryLow { --c: var(--bad); }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="logo">🎓</div>
    <div>
      <h1>Admission Predictor</h1>
      <p>AI-powered admission chance estimator (AdaBoost)</p>
    </div>
  </div>
  <button class="icon-btn" id="themeBtn" title="Toggle light / dark theme" aria-label="Toggle theme">🌙</button>
</header>

<main>
  <section class="card">
    <div class="toolbar">
      <span class="label">Quick presets:</span>
      <button class="pill" data-preset="strong">🌟 Strong applicant</button>
      <button class="pill" data-preset="average">📘 Average applicant</button>
      <button class="pill" data-preset="weak">🌱 Needs improvement</button>
      <button class="pill" data-preset="random">🎲 Random</button>
    </div>
    <form id="form" autocomplete="off"></form>
    <div class="actions">
      <button class="btn primary" id="predictBtn" type="button">Predict my chances</button>
      <button class="btn ghost" id="resetBtn" type="button">Reset</button>
    </div>
  </section>

  <aside>
    <section class="card" id="result">
      <div class="empty">
        <div class="big">🔮</div>
        <b>Your result will appear here</b>
        <p>Fill in the details (or pick a preset) and press <b>Predict my chances</b>.</p>
      </div>
    </section>
    <section class="card" id="histCard" style="display:none">
      <div class="hist-title"><b>Recent predictions</b><button id="clearHist">Clear</button></div>
      <div class="hist" id="hist"></div>
    </section>
    <p class="disclaimer">This is a statistical estimate from a machine-learning model, not an official decision.</p>
  </aside>
</main>
<div id="toast"></div>

<script>
const FIELDS = __FIELDS__;
const PRESETS = {
  strong:  { high_school_gpa: 3.9, sat_score: 1480, act_score: 33, ap_courses: 8, attendance_rate: 98, extracurricular_count: 6, volunteer_hours: 150, leadership_positions: 3, coding_projects: 5, online_certifications: 5, essay_score: 90, recommendation_score: 88, interview_score: 90, social_media_hours: 1.5 },
  average: { high_school_gpa: 3.4, sat_score: 1200, act_score: 26, ap_courses: 3, attendance_rate: 93, extracurricular_count: 3, volunteer_hours: 60, leadership_positions: 1, coding_projects: 2, online_certifications: 2, essay_score: 70, recommendation_score: 70, interview_score: 70, social_media_hours: 3 },
  weak:    { high_school_gpa: 2.3, sat_score: 900, act_score: 18, ap_courses: 0, attendance_rate: 78, extracurricular_count: 1, volunteer_hours: 10, leadership_positions: 0, coding_projects: 0, online_certifications: 0, essay_score: 45, recommendation_score: 45, interview_score: 45, social_media_hours: 6 }
};
const $ = (s, r = document) => r.querySelector(s);
const form = $("#form");

/* ---------- build form ---------- */
const groups = [...new Set(FIELDS.map(f => f.group))];
groups.forEach(g => {
  const wrap = document.createElement("div");
  wrap.className = "group";
  wrap.innerHTML = `<h3>${g}</h3><div class="grid"></div>`;
  const grid = $(".grid", wrap);
  FIELDS.filter(f => f.group === g).forEach(f => {
    const d = document.createElement("div");
    d.className = "field";
    d.innerHTML = `
      <label for="n_${f.key}">${f.label}<span>${f.unit}</span></label>
      <div class="row">
        <input type="range" id="r_${f.key}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.default}">
        <input class="num" type="number" id="n_${f.key}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.default}">
      </div>
      <div class="hint">${f.hint}</div>`;
    grid.appendChild(d);
    const r = $("#r_" + f.key, d), n = $("#n_" + f.key, d);
    const paint = () => r.style.setProperty("--p", ((r.value - f.min) / (f.max - f.min) * 100) + "%");
    r.addEventListener("input", () => { n.value = r.value; paint(); });
    n.addEventListener("input", () => {
      let v = parseFloat(n.value);
      if (!isNaN(v)) { r.value = Math.min(Math.max(v, f.min), f.max); paint(); }
    });
    n.addEventListener("blur", () => {
      let v = parseFloat(n.value);
      if (isNaN(v)) v = f.default;
      v = Math.min(Math.max(v, f.min), f.max);
      n.value = v; r.value = v; paint();
    });
    d._set = v => { n.value = v; r.value = v; paint(); };
    d.dataset.key = f.key;
    paint();
  });
  form.appendChild(wrap);
});

function setValues(obj) {
  FIELDS.forEach(f => {
    if (obj[f.key] !== undefined) $(`.field[data-key="${f.key}"]`)._set(obj[f.key]);
  });
}
function getValues() {
  const o = {};
  FIELDS.forEach(f => { o[f.key] = parseFloat($("#n_" + f.key).value); });
  return o;
}
function randomValues() {
  const o = {};
  FIELDS.forEach(f => {
    const steps = Math.round((f.max - f.min) / f.step);
    let v = f.min + Math.round(Math.random() * steps) * f.step;
    if (f.key === "family_income") v = 20000 + Math.round(Math.random() * 100) * 1000;
    if (f.key === "age") v = 17 + Math.round(Math.random() * 3);
    o[f.key] = +v.toFixed(2);
  });
  o.high_school_gpa = +(2 + Math.random() * 2).toFixed(2);
  o.sat_score = 800 + Math.round(Math.random() * 80) * 10;
  o.act_score = 15 + Math.round(Math.random() * 20);
  o.attendance_rate = 70 + Math.round(Math.random() * 30);
  o.ap_courses = Math.round(Math.random() * 8);
  o.extracurricular_count = Math.round(Math.random() * 7);
  o.volunteer_hours = Math.round(Math.random() * 40) * 5;
  o.leadership_positions = Math.round(Math.random() * 3);
  o.coding_projects = Math.round(Math.random() * 6);
  o.online_certifications = Math.round(Math.random() * 6);
  o.essay_score = 40 + Math.round(Math.random() * 55);
  o.recommendation_score = 40 + Math.round(Math.random() * 55);
  o.interview_score = 40 + Math.round(Math.random() * 55);
  o.social_media_hours = Math.round(Math.random() * 10) / 2;
  return o;
}

/* ---------- theme ---------- */
const themeBtn = $("#themeBtn");
function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  localStorage.setItem("theme", t);
  themeBtn.textContent = t === "dark" ? "☀️" : "🌙";
}
applyTheme(document.documentElement.getAttribute("data-theme"));
themeBtn.onclick = () => applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");

/* ---------- helpers ---------- */
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 2400);
}
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function countUp(el, to) {
  const start = performance.now(), dur = 1000;
  (function tick(now) {
    const k = Math.min((now - start) / dur, 1), e = 1 - Math.pow(1 - k, 3);
    el.textContent = (to * e).toFixed(1) + "%";
    if (k < 1) requestAnimationFrame(tick);
  })(start);
}

/* ---------- history ---------- */
let history = [];
try { history = JSON.parse(localStorage.getItem("history") || "[]"); } catch (e) {}
function renderHistory() {
  $("#histCard").style.display = history.length ? "block" : "none";
  $("#hist").innerHTML = history.map((h, i) => `
    <div class="hist-item tier-${h.tier}">
      <span>${esc(h.time)} · GPA ${h.gpa} · SAT ${h.sat}</span>
      <b>${h.p}%</b>
    </div>`).join("");
}
renderHistory();
$("#clearHist").onclick = () => { history = []; localStorage.removeItem("history"); renderHistory(); };

/* ---------- predict ---------- */
let lastReport = null;
async function predict() {
  const btn = $("#predictBtn");
  const vals = getValues();
  for (const f of FIELDS) {
    if (isNaN(vals[f.key])) { toast(`Please enter a valid number for "${f.label}"`); return; }
  }
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Analysing...';
  try {
    const res = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(vals) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");
    lastReport = data;
    showResult(data, vals);
    history.unshift({ p: data.probability, tier: data.tier, gpa: vals.high_school_gpa, sat: vals.sat_score, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) });
    history = history.slice(0, 5);
    try { localStorage.setItem("history", JSON.stringify(history)); } catch (e) {}
    renderHistory();
    if (window.innerWidth < 960) $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    toast(err.message || "Network error. Please try again.");
  } finally {
    btn.disabled = false; btn.textContent = "Predict my chances";
  }
}

function showResult(d) {
  const el = $("#result");
  el.className = "card fade-in tier-" + d.tier;
  const strengths = d.strengths.length ? d.strengths.map(s => `<span class="chip good">✓ ${esc(s.text)}</span>`).join("") : '<span class="hint">No standout strengths yet.</span>';
  const gaps = d.gaps.length ? d.gaps.map(s => `<span class="chip bad">↓ ${esc(s.text)}</span>`).join("") : '<span class="hint">Nothing below the typical level. Great job!</span>';
  const tips = d.tips.length ? d.tips.map(t => `<div class="tip"><span>${esc(t.text)}</span><b>+${t.gain}%</b></div>`).join("") : '<span class="hint">Your profile is already near its maximum potential.</span>';
  el.innerHTML = `
    <div class="verdict">
      <div class="gauge-wrap">
        <svg class="gauge" viewBox="0 0 120 120"><circle class="bg" cx="60" cy="60" r="54"/><circle class="fg" cx="60" cy="60" r="54"/></svg>
        <div class="gauge-text"><b id="pct">0%</b><small>chance of admission</small></div>
      </div>
      <span class="badge">${d.admitted ? "✅" : "⚠️"} ${esc(d.verdict)}</span>
      <h2>${esc(d.headline)}</h2>
      <p>${esc(d.summary)}</p>
    </div>
    <div class="sec"><h4>💪 What is working for you</h4><div class="chips">${strengths}</div></div>
    <div class="sec"><h4>📉 Where you can improve</h4><div class="chips">${gaps}</div></div>
    <div class="sec"><h4>🚀 Biggest ways to boost your chances</h4>${tips}</div>
    <div class="share">
      <button class="btn ghost" id="copyBtn">📋 Copy summary</button>
      <button class="btn ghost" id="printBtn">🖨️ Print</button>
    </div>`;
  requestAnimationFrame(() => {
    const fg = $(".gauge .fg", el);
    fg.style.strokeDashoffset = 339.29 * (1 - d.probability / 100);
  });
  countUp($("#pct", el), d.probability);
  $("#copyBtn").onclick = copySummary;
  $("#printBtn").onclick = () => window.print();
}

function copySummary() {
  if (!lastReport) return;
  const d = lastReport;
  const text = `Admission Prediction: ${d.probability}% chance. ${d.verdict}. ${d.headline}. ${d.summary}` +
    (d.tips.length ? ` Top suggestions: ${d.tips.map(t => t.text + " (+" + t.gain + "%)").join("; ")}.` : "");
  (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject())
    .then(() => toast("Summary copied to clipboard"))
    .catch(() => toast("Could not copy automatically"));
}

/* ---------- events ---------- */
$("#predictBtn").onclick = predict;
$("#resetBtn").onclick = () => { const o = {}; FIELDS.forEach(f => o[f.key] = f.default); setValues(o); toast("Form reset"); };
document.querySelectorAll("[data-preset]").forEach(b => b.onclick = () => {
  const p = b.dataset.preset;
  setValues(p === "random" ? randomValues() : PRESETS[p]);
  predict();
});
document.addEventListener("keydown", e => { if (e.key === "Enter" && e.target.classList.contains("num")) { e.preventDefault(); e.target.blur(); predict(); } });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)