import io, re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

st.set_page_config(page_title="Resume ↔ JD Analyzer", page_icon="🔎", layout="centered")

st.title("🔎 Resume ↔ Job Description Analyzer")

jd = st.text_area("Paste the Job Description", height=220, placeholder="Responsibilities, requirements, tech stack...")
uploaded = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

def clean(txt):
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9+#.\-\s]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return text

if st.button("Analyze", disabled=not (jd and uploaded)):
    resume_text = extract_text_from_pdf(uploaded)
    jd_c = clean(jd); res_c = clean(resume_text)

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words="english")
    X = vec.fit_transform([jd_c, res_c])
    score = cosine_similarity(X[0], X[1])[0,0]
    st.subheader(f"Match score: **{score*100:.1f}%**")

    # simple keyword gap check
    req_keywords = set([w for w in re.findall(r"[a-zA-Z0-9+#.\-]{3,}", jd_c) if w not in {"and","the","for","with","you","are"}])
    res_words = set(re.findall(r"[a-zA-Z0-9+#.\-]{3,}", res_c))
    missing = sorted([k for k in req_keywords if k not in res_words])[:40]

    st.markdown("### ✅ Suggestions")
    bullets = []
    if score < 0.7: bullets.append("Tailor the **summary** to echo the job description’s language.")
    if len(missing) > 0: bullets.append("Add these missing keywords where truthful: " + ", ".join(missing[:20]) + ("…" if len(missing)>20 else ""))
    if "python" in jd_c and "python" not in res_c: bullets.append("Show **Python** impact: projects, repos, metrics.")
    if "sql" in jd_c and "sql" not in res_c: bullets.append("Add an SQL bullet with query/report results.")
    if "ml" in jd_c and "ml" not in res_c: bullets.append("Mention models, data size, and evaluation metrics.")
    if bullets: st.write("\n".join([f"- {b}" for b in bullets]))
    else: st.write("- Looks like a strong match! Minor wording tweaks only to better match job description, while being truthful.")

    with st.expander("View extracted resume text (debug)"):
        st.write(resume_text[:6000])
