import io, re, json, textwrap
from collections import Counter, defaultdict

import streamlit as st
import pandas as pd

# Lightweight NLP & similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Resume file types
import PyPDF2
try:
    import docx  # python-docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

# -------------------- UI CONFIG --------------------
st.set_page_config(
    page_title="Resume ↔ Job Ad Matcher",
    page_icon="🧠",
    layout="wide"
)
st.title("🧠 Resume ↔ Job Ad Matcher")

st.caption(
    "Compare any resume to any job ad. We score the match and suggest concrete improvements."
)

# -------------------- ROLE PRESETS (OPTIONAL) --------------------
ROLE_PRESETS = {
    "Generic / Other": {
        # No fixed taxonomy – uses job ad keywords + impact only.
        "core": [],
        "ml": [],
        "viz": [],
        "data": [],
        "quality": [],
        "ops": [],
    },
    "Data Scientist": {
        "core": [
            "python","pandas","numpy","scikit-learn","sklearn","statistics",
            "sql","experiment","ab testing","hypothesis","regression",
            "classification","cross-validation","feature engineering","pipelines",
            "model evaluation","roc auc","precision","recall","f1",
            "confusion matrix","xgboost","lightgbm","random forest","time series",
            "forecasting","causal","uplift","bayesian"
        ],
        "ml": [
            "xgboost","lightgbm","random forest","svm","logistic regression",
            "linear regression","kmeans","dbscan","prophet","arima","lstm",
            "transformer","nlp","llm","bert","shap","lime","explainability","fairness"
        ],
        "viz": ["power bi","tableau","plotly","matplotlib","seaborn","looker"],
        "cloud": ["aws","gcp","azure","databricks","vertex ai","sagemaker","fabric"],
    },
    "ML Engineer": {
        "core": [
            "python","pandas","numpy","pytorch","tensorflow","mlops","docker",
            "kubernetes","api","fastapi","looker","airflow","mlflow","wandb",
            "dvc","feature store","model registry","monitoring","inference",
            "latency","throughput"
        ],
        "ml": [
            "pytorch","tensorflow","onnx","torchserve","triton","hf transformers",
            "distillation","quantization","vector db","faiss","chroma","weaviate",
            "rag","retrieval","embedding"
        ],
        "cloud": [
            "aws","gcp","azure","kubernetes","terraform","helm","ci/cd",
            "github actions","cloud run","vertex ai","sagemaker"
        ],
        "data": ["spark","databricks","bigquery","snowflake","kafka"],
    },
    "Data Engineer": {
        "core": [
            "sql","data modeling","etl","elt","pipelines","airflow","dbt",
            "orchestration","dimensional modeling","star schema","warehouse",
            "lakehouse","spark","databricks","kafka","flink"
        ],
        "cloud": [
            "gcp","bigquery","pubsub","dataflow","aws","glue","redshift",
            "kinesis","azure","synapse","fabric"
        ],
        "quality": ["testing","great expectations","observability","metadata","data lineage"],
        "ops": ["docker","kubernetes","terraform","ci/cd"],
    },
    "Data Analyst": {
        "core": [
            "sql","excel","power bi","tableau","looker","dashboards","storytelling",
            "kpi","ab testing","hypothesis","segmentation","cohort","retention"
        ],
        "stats": ["statistics","anova","regression","forecast","time series","seasonality"],
        "py": ["python","pandas","numpy","plotly","matplotlib","seaborn"],
        "viz": ["power bi","tableau","looker","dashboards"],
    }
}

# Default component weights
DEFAULT_WEIGHTS = {
    "tfidf": 0.40,
    "core": 0.25,
    "ml": 0.15,
    "viz_or_data_or_ops": 0.10,  # depends on role
    "impact": 0.10,              # metrics/impact verbs
}

IMPACT_PATTERNS = [
    r"\b\d{1,3}%\b",                      # percentages
    r"\b\d{1,3}(?:\.\d+)?\s*(?:k|m|b)\b", # 10k, 3.2M
    r"\b(?:reduced|improved|increased|decreased|cut|boosted)\b",
    r"\b(?:latency|throughput|accuracy|f1|precision|recall|auc|rmse|mae|revenue|cost)\b",
]

STOP = {
    "and","the","for","with","you","are","to","of","in","on","a","an","by",
    "at","as","or","vs","be","is","was","were","from","that","this"
}

# -------------------- HELPERS --------------------
def read_pdf(file) -> str:
    pdf = PyPDF2.PdfReader(io.BytesIO(file.read()))
    return "\n".join(page.extract_text() or "" for page in pdf.pages)

def read_docx(file) -> str:
    if not HAS_DOCX:
        return ""
    d = docx.Document(io.BytesIO(file.read()))
    return "\n".join(p.text for p in d.paragraphs)

def clean(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9+#.\-_/()%\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def tokenize(txt: str):
    return [w for w in re.findall(r"[a-z0-9+#.\-_/()%]{2,}", txt) if w not in STOP]

def tfidf_cosine(a: str, b: str) -> float:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    X = vec.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0, 0])

def count_hits(text: str, terms: list[str]) -> tuple[int, list[str]]:
    found = []
    T = " " + text + " "
    for t in set(terms):
        if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", T):
            found.append(t)
    return len(found), sorted(found)

def find_impact_signals(text: str) -> int:
    return sum(len(re.findall(p, text)) for p in IMPACT_PATTERNS)

def unique_missing(required: list[str], present: list[str]) -> list[str]:
    pres = set(present)
    return [t for t in required if t not in pres]

# -------------------- SIDEBAR --------------------
st.sidebar.header("Target profile (optional)")
role = st.sidebar.selectbox(
    "Which profile are you aiming for?",
    list(ROLE_PRESETS.keys()),
    index=0
)
bundle = ROLE_PRESETS[role]

# choose secondary category depending on role
if role in ("Data Scientist", "Data Analyst"):
    secondary_key = "viz"
elif role == "ML Engineer":
    secondary_key = "data"
elif role == "Data Engineer":
    secondary_key = "quality"
else:
    secondary_key = "other"  # Generic / Other

st.sidebar.markdown("---")
st.sidebar.header("Scoring weights")

w = DEFAULT_WEIGHTS.copy()
w["viz_or_data_or_ops"] = st.sidebar.slider(
    "Weight: secondary category",
    0.0, 0.30, w["viz_or_data_or_ops"], 0.01
)
w["tfidf"] = st.sidebar.slider(
    "Weight: JD ↔ resume similarity (TF-IDF)",
    0.0, 0.60, w["tfidf"], 0.01
)
w["core"] = st.sidebar.slider(
    "Weight: core skills coverage",
    0.0, 0.60, w["core"], 0.01
)
w["ml"] = st.sidebar.slider(
    "Weight: advanced/ML coverage",
    0.0, 0.60, w["ml"], 0.01
)
w["impact"] = st.sidebar.slider(
    "Weight: impact signals (numbers, %)",
    0.0, 0.30, w["impact"], 0.01
)

st.sidebar.caption("Weights are normalized automatically.")

st.sidebar.markdown("---")
st.sidebar.header("Resume input")

input_mode = st.sidebar.radio(
    "How do you want to provide your resume?",
    ["Upload file", "Paste text"],
    index=0
)

resume_file = None
resume_text = None

if input_mode == "Upload file":
    resume_file = st.sidebar.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"] if HAS_DOCX else ["pdf"]
    )
else:
    resume_text = st.sidebar.text_area(
        "Paste your resume / CV text",
        height=260,
        placeholder="Paste your CV here (can be in any language)..."
    )

# -------------------- JOB DESCRIPTION --------------------
st.subheader("Job Description")
jd = st.text_area(
    "Paste the Job Description / Job Ad",
    height=220,
    placeholder="Paste responsibilities, requirements, tech stack, soft skills, etc..."
)

can_analyze = bool(
    jd and (
        (input_mode == "Upload file" and resume_file is not None)
        or (input_mode == "Paste text" and resume_text and resume_text.strip())
    )
)
analyze = st.button("Analyze", type="primary", disabled=not can_analyze)

# -------------------- MAIN LOGIC --------------------
if analyze:
    # 1) Read & clean resume
    if input_mode == "Upload file":
        ext = (resume_file.name.split(".")[-1] or "").lower()
        if ext == "pdf":
            resume_raw = read_pdf(resume_file)
        elif ext == "docx" and HAS_DOCX:
            resume_raw = read_docx(resume_file)
        else:
            st.error("Unsupported file type. Please upload a PDF" + (" or DOCX" if HAS_DOCX else ""))
            st.stop()
    else:
        resume_raw = resume_text

    jd_c = clean(jd)
    res_c = clean(resume_raw or "")

    if not res_c:
        st.error("Could not read any text from your resume. Try pasting the text instead.")
        st.stop()

    # 2) TF-IDF similarity (content alignment)
    sim = tfidf_cosine(jd_c, res_c)

    # 3) Skills/taxonomy coverage
    def get_terms(key):
        return [t for t in bundle.get(key, [])]

    core_terms = get_terms("core")
    ml_terms = get_terms("ml")
    sec_terms = (
        get_terms(secondary_key)
        or get_terms("viz")
        or get_terms("data")
        or get_terms("ops")
        or []
    )

    core_hits_n, core_hits = count_hits(res_c, core_terms)
    ml_hits_n, ml_hits = count_hits(res_c, ml_terms)
    sec_hits_n, sec_hits = count_hits(res_c, sec_terms)

    # 4) Impact signals (numbers, % changes, metrics verbs)
    impact_score = min(1.0, find_impact_signals(res_c) / 6.0)  # saturate around ~6 signals

    # 5) Normalize component scores
    def norm(n_found, n_total):
        return 0.0 if n_total == 0 else n_found / n_total

    s_core = norm(core_hits_n, len(set(core_terms)))
    s_ml = norm(ml_hits_n, len(set(ml_terms)))
    s_sec = norm(sec_hits_n, len(set(sec_terms)))
    s_tfidf = sim

    # 6) Adjust weights if no taxonomy (Generic / Other)
    has_taxonomy = (
        len(core_terms) > 0 or
        len(ml_terms) > 0 or
        len(sec_terms) > 0
    )
    if not has_taxonomy:
        # Only similarity + impact matter for generic roles
        w["core"] = 0.0
        w["ml"] = 0.0
        w["viz_or_data_or_ops"] = 0.0

    # 7) Weighted overall score (normalized weights)
    wsum = sum(w.values()) or 1.0
    weights = {k: v / wsum for k, v in w.items()}

    overall = (
        weights["tfidf"] * s_tfidf +
        weights["core"] * s_core +
        weights["ml"] * s_ml +
        weights["viz_or_data_or_ops"] * s_sec +
        weights["impact"] * impact_score
    )

    if not has_taxonomy:
        st.info(
            "You selected a generic profile or a role without a preset taxonomy. "
            "The score is based on job ad similarity + impact signals."
        )

    # 8) Missing items against JD and against role preset
    jd_tokens = [t for t in tokenize(jd_c) if len(t) >= 3]
    top_jd_counts = Counter(jd_tokens)
    jd_top = [term for term, _ in top_j_counts.most_common(80)]
    _, present_from_jd = count_hits(res_c, jd_top)
    missing_vs_jd = unique_missing(jd_top, present_from_jd)[:25]

    missing_core = unique_missing(core_terms, core_hits)[:20]
    missing_ml = unique_missing(ml_terms, ml_hits)[:20]
    missing_sec = unique_missing(sec_terms, sec_hits)[:20]

    # -------------------- OUTPUT --------------------
    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.subheader(f"Overall match: **{overall * 100:.1f}%**")
        st.caption(
            "Hybrid score using job ad similarity, skill coverage (if available), and impact signals."
        )

        # Short natural-language summary
        match_label = (
            "Strong" if overall >= 0.75
            else "Decent" if overall >= 0.5
            else "Needs more tailoring"
        )
        st.markdown(f"**Overall impression:** {match_label} match for this job ad.")

        # Score breakdown table
        parts = pd.DataFrame(
            [
                ["JD ↔ resume similarity (TF-IDF)", f"{s_tfidf * 100:.1f}%", weights["tfidf"]],
                ["Core skills coverage", f"{s_core * 100:.1f}%", weights["core"]],
                ["Advanced/ML coverage", f"{s_ml * 100:.1f}%", weights["ml"]],
                [f"Secondary ({secondary_key})", f"{s_sec * 100:.1f}%", weights["viz_or_data_or_ops"]],
                ["Impact signals", f"{impact_score * 100:.1f}%", weights["impact"]],
            ],
            columns=["Component", "Score", "Weight (normalized)"]
        )
        st.dataframe(parts, use_container_width=True)

        st.markdown("#### What you already cover (taxonomy terms)")
        st.write(", ".join(sorted(set(core_hits + ml_hits + sec_hits))) or "—")

    with col2:
        st.markdown("#### Top missing vs this Job Ad")
        st.write(", ".join(missing_vs_jd) or "—")

        st.markdown("#### Missing by category")
        tabs = st.tabs(["Core", "Advanced/ML", secondary_key.capitalize()])
        with tabs[0]:
            st.write(", ".join(missing_core) or "—")
        with tabs[1]:
            st.write(", ".join(missing_ml) or "—")
        with tabs[2]:
            st.write(", ".join(missing_sec) or "—")

    # -------------------- SUGGESTIONS --------------------
    st.markdown("### ✅ Suggestions to improve your resume for THIS job")

    sugg = []

    if s_core < 0.7 and len(core_terms) > 0:
        sugg.append(
            "Add a **Core Skills** line that mirrors the job ad wording "
            "(e.g. Python, SQL, statistics, experimentation)."
        )
    if s_ml < 0.6 and role in ("Data Scientist", "ML Engineer") and len(ml_terms) > 0:
        sugg.append(
            "Include **model details** (algorithms, data size, features, evaluation metrics) "
            "for 2–3 projects that are closest to this job."
        )
    if impact_score < 0.5:
        sugg.append(
            "Quantify **impact** with numbers (e.g. *+9% AUC, −30% latency, +€120k revenue, −15% cost*)."
        )
    if "sql" in jd_c and "sql" not in res_c:
        sugg.append(
            "Add a bullet about **SQL** if true (complex joins, CTEs, window functions, query performance)."
        )
    if role == "Data Scientist" and ("ab testing" in jd_c or "experiment" in jd_c) and "ab" not in res_c:
        sugg.append(
            "Include an **A/B testing** bullet (design, sample size, power, statistical tests)."
        )
    if role == "ML Engineer" and "mlops" in jd_c and "mlops" not in res_c:
        sugg.append(
            "Add an **MLOps** bullet (pipelines, model registry, CI/CD, monitoring in production)."
        )
    if role == "Data Engineer" and "dbt" in jd_c and "dbt" not in res_c:
        sugg.append(
            "Mention **dbt** models/tests and orchestration (e.g. Airflow)."
        )

    # Pull 8 most important JD terms missing
    if missing_vs_jd:
        sugg.append(
            "Weave in missing job ad terms where truthful, especially in your summary and recent roles: "
            + ", ".join(missing_vs_jd[:10])
        )

    if sugg:
        for s in sugg:
            st.write(f"- {s}")
    else:
        st.write(
            "- This is already a strong match. Focus on phrasing bullets to echo the job ad wording."
        )

    # -------------------- AUTO BULLET GENERATOR --------------------
    st.markdown("### ✍️ Draft resume bullets (edit & copy)")

    def bullet(role: str, term: str) -> str:
        return {
            "Generic / Other": f"Delivered results around **{term}**, demonstrating ownership, collaboration, and measurable impact (e.g. X% improvement / Y cost reduction).",
            "Data Scientist": f"Built and shipped **{term}** model(s) on real data, improving key metric by X% (A/B tested; n≈N; 95% CI).",
            "ML Engineer":    f"Productionized **{term}** with APIs and CI/CD; cut p95 latency by X% and reduced infra cost by Y%.",
            "Data Engineer":  f"Modeled **{term}** in a warehouse/lakehouse; automated ingestion with orchestration and improved data freshness to <15 min.",
            "Data Analyst":   f"Delivered **{term}** dashboard/analysis tracking KPIs; drove +X% uplift via insights and experiments.",
        }.get(role, f"Delivered impact using **{term}**, with clear metrics and ownership.")

    picks = (missing_ml or missing_core or missing_sec)[:5]
    if not picks:
        if role == "Data Scientist":
            picks = ["classification", "xgboost", "shap", "sql", "power bi"]
        elif role == "ML Engineer":
            picks = ["mlops", "pytorch", "inference", "monitoring", "docker"]
        elif role == "Data Engineer":
            picks = ["dbt", "spark", "airflow", "data quality", "etl"]
        elif role == "Data Analyst":
            picks = ["dashboard", "kpi", "cohort", "segmentation", "sql"]
        else:
            picks = ["stakeholder management", "ownership", "delivery", "communication", "impact"]

    text = "\n".join([f"• {bullet(role, t)}" for t in picks])
    st.text_area("Draft bullets:", value=text, height=180)

    # -------------------- EXPORTS --------------------
    st.markdown("### ⬇️ Export")

    # CSV of missing terms
    missing_df = pd.DataFrame({
        "missing_vs_jd": pd.Series(missing_vs_jd),
        "missing_core": pd.Series(missing_core),
        "missing_ml": pd.Series(missing_ml),
        f"missing_{secondary_key}": pd.Series(missing_sec),
    })
    st.download_button(
        "Download missing keywords (CSV)",
        data=missing_df.to_csv(index=False),
        file_name="missing_keywords.csv",
        mime="text/csv"
    )

    # JSON report
    report = {
        "role": role,
        "overall": round(overall, 4),
        "components": {
            "tfidf": s_tfidf,
            "core": s_core,
            "ml": s_ml,
            secondary_key: s_sec,
            "impact": impact_score,
        },
        "present": sorted(set(core_hits + ml_hits + sec_hits)),
        "missing": {
            "vs_jd": missing_vs_jd,
            "core": missing_core,
            "ml": missing_ml,
            secondary_key: missing_sec,
        },
        "suggestions": sugg,
    }
    st.download_button(
        "Download JSON report",
        data=json.dumps(report, indent=2),
        file_name="report.json",
        mime="application/json"
    )

    with st.expander("View extracted resume text (debug)"):
        st.write(resume_raw[:6000])
