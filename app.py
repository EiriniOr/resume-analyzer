import io
import re
import json
from collections import Counter
import re

import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import PyPDF2
try:
    import docx  # python-docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

# -------------------- TIPS --------------------

ATS_TIPS_TEXT = """

1. **Mirror the job ad** – Use the same key skills and phrases (truthfully) in your CV, especially in your summary and recent roles.  
2. **Use simple formatting** – One column, no tables or graphics, standard headings (Experience, Education, Skills) so ATS can parse correctly.  
3. **Match titles where you can** – Align your title/summary with the target role (e.g. “Product Owner / Data Scientist”) if it reflects your experience.  
4. **Put important keywords high** – Prioritise the most relevant tools, domains and responsibilities in the top third of your CV.  
5. **Add a clear skills section** – List core tools, methods and languages in a dedicated Skills/Kompetenser section for easy matching.  
6. **Tailor for each application** – Reorder bullets, add/remove details so your CV specifically reflects *this* job description.  
7. **Avoid keyword stuffing** – Repeat key terms a few times in context, but don’t dump buzzwords; humans still read it.  
8. **Use ATS-friendly files** – Prefer .docx or a text-based PDF, not image-only or heavily designed templates.
"""


# -------------------- UI CONFIG --------------------
st.set_page_config(
    page_title="ATS-style Job Match Scorer",
    layout="wide"
)

st.title("ATS-style job match scorer")
st.caption(
    "How high would you relatively score if you applied for an advertised position? Disclaimer, this is an estimation and a real Applicant Tracking System may differ."
)

# -------------------- CONFIG / CONSTANTS --------------------
STOP_EN = {
    # articles, pronouns, auxiliaries
    "a","an","the","and","or","but","if","then","than","so","because",
    "i","you","he","she","it","we","they","me","him","her","us","them",
    "my","your","yours","our","ours","their","theirs","his","hers","its",
    "this","that","these","those","there","here","where","when","why","how",
    "any","some","no","not","only","just","very",

    # common verbs/aux
    "am","is","are","was","were","be","been","being",
    "have","has","had","having","do","does","did","doing",
    "will","would","shall","should","can","could","may","might","must",

    # prepositions / linkers
    "to","of","in","on","at","by","for","from","with","about","into","over",
    "under","between","through","during","before","after","up","down","out",
    "off","across","within","without",

    # misc
    "as","such","also","again","more","most","many","much","few","each",
    "other","another","same","own","too"
}

STOP_SV = {
    # artiklar, pronomen
    "en","ett","den","det","de","dom","jag","du","han","hon","vi","ni",
    "mig","dig","honom","henne","oss","er","dem",
    "min","mitt","mina","din","ditt","dina","sin","sitt","sina","vår","vårt",
    "våra","er","ert","era","deras",
    "denna","detta","dessa","här","där","sådan","sådant","sådana",

    # vanliga verb/hjälpverb
    "är","var","vara","blir","blev","bli",
    "har","hade","haft","ha",
    "gör","gjorde","göra",
    "kan","kunde","ska","skall","skulle","får","fick","måste","bör","borde",

    # bindeord / prepositioner
    "och","eller","men","utan","fast","för",
    "att","som","om","när","medan","innan","efter","eftersom",
    "i","på","för","till","från","med","utan","över","under","mellan",
    "genom","hos","inom","utanför","innan","efter","under",
    "inte","ingen","inget","inga",

    # övrigt
    "också","så","då","också","redan","bara","mycket","många","få","några",
    "alla","varje","någon","något","några","samma","annan","andra","kanske"
}

SOFT_SKILLS = {
    "English": {
        # existing
        "communication", "teamwork", "collaboration", "leadership", "problem solving",
        "critical thinking", "adaptability", "flexibility", "time management",
        "organisation", "organization", "stakeholder management", "ownership",
        "initiative", "creativity", "empathy", "customer focus", "detail oriented",
        "detail-oriented", "self-motivated", "proactive", "analytical",
        "negotiation", "presentation",

        # more communication / collaboration
        "communication skills", "strong communication skills",
        "interpersonal skills", "relationship building",
        "cross-functional collaboration", "cross functional collaboration",
        "stakeholder communication", "stakeholder engagement",
        "team player", "supportive",

        # thinking / analysis / strategy
        "analytical thinking", "strategic thinking", "conceptual thinking",
        "problem-solving skills", "data-driven mindset", "data driven mindset",
        "decision making", "decision-making",

        # ownership / drive
        "accountability", "sense of ownership", "end-to-end ownership",
        "self-driven", "self driven", "autonomous", "independent",
        "self starter", "self-starter",

        # planning / structure
        "planning", "prioritisation", "prioritization",
        "multitasking", "attention to detail", "organized", "organised",

        # customer / user orientation
        "customer-centric", "customer centric",
        "user-centric", "user centric",
        "service minded", "service-minded",

        # influence / facilitation
        "influencing", "influence skills",
        "conflict resolution", "mediation",
        "facilitation", "workshop facilitation",
        "coaching", "mentoring",

        # results / solutions / mindset
        "result oriented", "result-oriented",
        "outcome driven", "outcome-driven",
        "solution oriented", "solution-oriented",
        "growth mindset", "continuous improvement",
        "learning mindset", "curiosity", "open-minded",

        # resilience / change
        "resilience", "stress tolerance",
        "change readiness", "change oriented", "change-oriented"
    },

    "Swedish": {
        # existing
        "kommunikation", "samarbete", "lagarbete", "ledarskap", "problemlösning",
        "kritiskt tänkande", "anpassningsförmåga", "flexibilitet", "tidsplanering",
        "struktur", "självständighet", "initiativförmåga", "kundfokus",
        "analytisk", "noggrann", "kommunikativ",

        # kommunikation / relationer
        "kommunikationsförmåga", "stark kommunikationsförmåga",
        "interpersonella färdigheter", "relationsskapande", "relationsbyggande",
        "teamkänsla", "lagspelare",

        # analys / strategi
        "analytiskt tänkande", "strategiskt tänkande",
        "problemlösningsförmåga", "dataorienterad", "dataorienterat arbetssätt",

        # ägarskap / ansvar / driv
        "ansvarstagande", "eget ansvar", "ägarskap",
        "självständig", "driv", "driven", "engagerad",
        "initiativrik", "högt engagemang",

        # struktur / planering
        "strukturerad", "planeringsförmåga", "prioriteringsförmåga",
        "förmåga att hantera flera uppgifter", "förmåga att hålla deadlines",

        # kund / användare
        "kundorienterad", "kundinriktad",
        "serviceinriktad", "serviceorienterad",
        "användarfokus", "användarcentrerad",

        # påverkan / facilitering
        "påverkansförmåga", "förhandlingsförmåga",
        "konflikthantering", "facilitering", "workshopledare",
        "coachande", "coachande ledarskap", "handledning",

        # resultat / lösningar / kvalitet
        "lösningsorienterad",
        "resultatorienterad", "resultatinriktad", "målinriktad",
        "kvalitetsmedveten", "hög kvalitet", "noggrannhet",

        # personlig stil / mindset
        "lyhörd", "ödmjuk", "pedagogisk", "pedagogisk förmåga",
        "flexibel", "förändringsbenägen", "förändringsvillig",
        "lärande mindset", "nyfiken"
    }
}



def get_stopwords(language: str):
    return STOP_EN if language == "English" else STOP_SV



IMPACT_PATTERNS = [
    r"\b\d{1,3}%\b",                      # percentages
    r"\b\d{1,3}(?:\.\d+)?\s*(?:k|m|b)\b", # 10k, 3.2M
    r"\b(?:reduced|improved|increased|decreased|cut|boosted|saved|grew)\b",
    r"\b(?:revenue|cost|profit|latency|throughput|accuracy|conversion|kpi|sales)\b",
]

# weights for components 
DEFAULT_WEIGHTS = {
    "similarity": 0.25,
    "keywords": 0.45,
    "soft_skills": 0.20,
    "impact": 0.10,
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

def clean(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    # keep unicode word chars (åäö etc.) + some punctuation
    text = re.sub(r"[^\w+#.\-_/()%\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_for_match(word: str, language: str) -> str:
    """
    Very lightweight 'stemming' so that singular/plural forms
    are treated as the same for matching/coverage.
    """
    w = word.lower()

    if language == "English":
        # e.g. 'companies' -> 'company'
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"

        # e.g. 'processes' -> 'process'
        if w.endswith("ses") and len(w) > 4:
            return w[:-2]

        # generic plural: 'models' -> 'model', but keep 'class', 'process'
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            return w[:-1]

        return w

    # Swedish – very rough, but good enough for 'system'/'systemen' etc.
    else:
        # longest suffixes first
        suffixes = [
            "arna", "erna", "orna",       # t.ex. systemen -> system, kurserna -> kurs
            "ande", "ende",
            "heten", "heter", "het",
            "na", "en", "et",
            "ar", "er", "or", "n",
        ]
        for suf in suffixes:
            if w.endswith(suf) and len(w) > len(suf) + 1:
                return w[:-len(suf)]
        return w


def tokenize(text: str, language: str):
    if not text:
        return []
    stop = get_stopwords(language)

    # Remove common punctuation so "word." / "word," / "\"word\"" → "word"
    # (but keep +, #, %, _, /, (), etc. for tech terms)
    text = re.sub(r"[.,;:!?\"“”‘’]", " ", text)

    # Build tokens without dots so we don't get "word."
    raw_tokens = re.findall(
        r"[\w#+\-_/()%]{2,}",  # note: '.' removed here
        text.lower(),
        flags=re.UNICODE,
    )

    return [w for w in raw_tokens if w not in stop]


def tfidf_cosine(a: str, b: str, language: str) -> float:
    stop_words = "english" if language == "English" else None
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        stop_words=stop_words,
    )
    X = vec.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0, 0])

def count_hits(text: str, terms: list[str]) -> tuple[int, list[str]]:
    if not text or not terms:
        return 0, []
    text = text.lower()
    terms = [t.lower() for t in terms]
    T = " " + text + " "
    found = []
    for t in set(terms):
        if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", T):
            found.append(t)
    return len(found), sorted(found)

def unique_missing(required: list[str], present: list[str]) -> list[str]:
    pres = set(present)
    return [t for t in required if t not in pres]

def find_impact_signals(text: str) -> int:
    if not text:
        return 0
    return sum(len(re.findall(p, text.lower())) for p in IMPACT_PATTERNS)

def detect_sections(text: str) -> dict:
    text_low = text.lower()
    present = {}
    for name, hints in SECTION_HINTS.items():
        present[name] = any(h in text_low for h in hints)
    return present

# -------------------- SIDEBAR: INPUT --------------------
st.sidebar.markdown(
    """
    <div style="
        border-left: 4px solid #1f6feb;
        background-color: rgba(15, 23, 42, 0.25);
        padding: 0.6rem 0.8rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
    ">
      <strong>Instructions</strong><br><br>
      • Upload your CV and paste the job ad<br>
      • Get an ATS-style match score<br>
      • See missing keywords and receive suggestions<br>
      • Weave the missing keywords/skills into your CV yourself, or via an LLM and improve your ATS-style compatibility for that specific role
    </div>
    """,
    unsafe_allow_html=True,
)


st.sidebar.header("Language")

LANGUAGE = st.sidebar.selectbox(
    "Choose the language of the ad & CV",
    ["English", "Swedish"],
    index=0
)

st.sidebar.header("Input your CV")
input_mode = st.sidebar.radio(
    "CV input mode",                  # not shown
    ["Upload file", "Paste text"],
    index=0,
    label_visibility="collapsed",
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
        "Paste your resume / CV",
        height=260,
        placeholder="Paste your CV here..."
    )

st.subheader("Insert the job advertisement information:")
job_title = st.text_input("Job title (optional, for context)", placeholder="e.g. Data Scientist, Marketing Specialist, Nurse...")
jd = st.text_area(
    "Paste the Job Description",
    height=260,
    placeholder="Paste responsibilities, requirements, skills, expectations, etc..."
)

can_analyze = bool(
    jd and (
        (input_mode == "Upload file" and resume_file is not None)
        or (input_mode == "Paste text" and resume_text and resume_text.strip())
    )
)
analyze = st.button("Analyze match", type="primary", disabled=not can_analyze)

# -------------------- MAIN LOGIC --------------------
if analyze:
    # 1) Get resume text
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

    if not resume_raw or not resume_raw.strip():
        st.error("Could not read any text from your CV. Try pasting the text instead.")
        st.stop()

    jd_clean = clean(jd)
    cv_clean = clean(resume_raw)

    # 2) TF-IDF similarity
    similarity = tfidf_cosine(jd_clean, cv_clean, LANGUAGE)


    # 3) Job-ad keywords (generic)
    jd_tokens = [t for t in tokenize(jd_clean, LANGUAGE) if len(t) >= 3]
    top_jd_counts = Counter(jd_tokens)
    jd_keywords = [term for term, _ in top_jd_counts.most_common(80)]

    # --- plural-insensitive keyword coverage ---
    cv_tokens = tokenize(cv_clean, LANGUAGE)

    def normalize_list(words: list[str]) -> list[str]:
        return [normalize_for_match(w, LANGUAGE) for w in words if w]

    jd_keywords_norm = normalize_list(jd_keywords)
    cv_norm_set = {normalize_for_match(w, LANGUAGE) for w in cv_tokens}

    present_keywords_norm = sorted({w for w in jd_keywords_norm if w in cv_norm_set})
    missing_keywords_norm = unique_missing(jd_keywords_norm, present_keywords_norm)


    # present vs missing keywords
    _, present_keywords = count_hits(cv_clean, jd_keywords)
    missing_keywords = unique_missing(jd_keywords, present_keywords)

    # 4) Soft skills coverage (based on job ad)
    jd_soft_vocab = SOFT_SKILLS[LANGUAGE]
    jd_soft = [s for s in jd_soft_vocab if s in jd_clean]
    _, present_soft = count_hits(cv_clean, jd_soft)
    missing_soft = unique_missing(jd_soft, present_soft)


    # 5) Impact signals
    impact_raw = find_impact_signals(cv_clean)
    impact_score = min(1.0, impact_raw / 6.0)  # saturate around ~6 signals

    # 6) Normalised component scores
    def norm_coverage(present: list[str], total: list[str]) -> float:
        if not total:
            return 0.0
        return len(set(present)) / len(set(total))

    s_keywords = norm_coverage(present_keywords_norm, jd_keywords_norm)

    s_soft = norm_coverage(present_soft, jd_soft)
    s_similarity = similarity

    # 7) Weighted overall score (0–1)
    w = DEFAULT_WEIGHTS.copy()
    wsum = sum(w.values())
    weights = {k: v / wsum for k, v in w.items()}

    overall = (
        weights["similarity"]   * s_similarity +
        weights["keywords"]     * s_keywords +
        weights["soft_skills"]  * s_soft +
        weights["impact"]       * impact_score
    )
    overall_pct = overall * 100

    # -------------------- OUTPUT --------------------
    title_display = job_title or "this role"

    # Decide label + color based on score
    if overall >= 0.70:
        label = "Strong match"
        color = "#22c55e"   # green
    elif overall >= 0.50:
        label = "Good match"
        color = "#a3e635"   # yellow-green
    else:
        label = "Needs improvement, follow suggestions!"
        color = "#f97316"   # orange / red

    # ----- 1. Overall match -----
    st.markdown("## 1. Overall match")

    st.markdown(
        f"""
        <div style="font-size: 1.1rem; margin-bottom: 0.3rem;">
        Match score for <strong>{title_display}</strong>:
        <span style="color: {color}; font-weight: 700;">
            {overall_pct:.1f}%
        </span>
        </div>
        <div style="font-size: 0.95rem; color: {color}; margin-bottom: 0.8rem;">
        ATS-style impression: <strong>{label}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- 2. Score breakdown -----
    st.markdown("## 2. Score breakdown")

    parts = pd.DataFrame(
        [
            ["Similarity (content overlap)",      f"{s_similarity * 100:.1f}%", weights["similarity"]],
            ["Job-ad keyword coverage",          f"{s_keywords   * 100:.1f}%", weights["keywords"]],
            ["Soft skills from job ad present",  f"{s_soft       * 100:.1f}%", weights["soft_skills"]],
            ["Impact signals (numbers, %)",      f"{impact_score * 100:.1f}%", weights["impact"]],
        ],
        columns=["Component", "Score", "Weight (normalized)"]
    )
    st.dataframe(parts, use_container_width=True)

    # ----- 3. Keyword coverage -----
    st.markdown("## 3. Keyword coverage")

    st.markdown("#### 3.1 Keywords already in your CV (from the job ad)")
    st.write(", ".join(sorted(set(present_keywords_norm))) or "—")

    st.markdown("#### 3.2 Top missing keywords (from this job ad)")
    st.write(", ".join(missing_keywords_norm[:30]) or "—")

    # ----- 4. Soft skills -----
    if jd_soft:
        st.markdown("## 4. Soft skills")

        st.markdown("#### 4.1 Soft skills the job ad explicitly mentions")
        st.write(", ".join(jd_soft))

        st.markdown("#### 4.2 Soft skills missing in your CV")
        st.write(", ".join(missing_soft) or "—")


    # -------------------- SUGGESTIONS --------------------
    st.markdown("### Suggestions to improve your score for THIS job application:")

    suggestions = []

    # similarity
    if s_similarity < 0.5:
        suggestions.append(
            "Rewrite your **summary and recent experience** to echo the job ad language "
            "(use similar phrases for responsibilities, tools and outcomes)."
        )

    # keywords
    if s_keywords < 0.7 and missing_keywords:
        suggestions.append(
            "Add some of the **most important missing keywords** (if they are true for you), "
            "especially in your summary and in the top 1–2 roles: "
            + ", ".join(missing_keywords[:10])
        )

    # soft skills
    if jd_soft and s_soft < 0.7:
        suggestions.append(
            "The job ad values certain **soft skills**. Add bullet points that *show* these skills in action "
            f"(e.g. situations where you used {', '.join(missing_soft[:5])})."
        )

    # impact
    if impact_score < 0.2:
        suggestions.append(
            "Add **numbers** to your bullets where possible: percentages, money saved/earned, time saved, "
            "customers served, etc. (e.g. *\"Increased conversion by 12%\"*, *\"Reduced processing time by 30%\"*)."
        )


    for s in suggestions:
        st.write(f"- {s}")

    with st.expander("Quick tips how to prepare your CV:"):
        st.markdown(ATS_TIPS_TEXT)


st.markdown(
    """
    <hr style="margin-top: 2rem; margin-bottom: 0.5rem;">
    <div style="font-size: 0.85rem; color: #64748b; text-align: center;">
      Built by <strong>Eirini Ornithopoulou (2025)</strong> ·
      <a href="https://eirini-portfolio-aer3.vercel.app/" target="_blank">
        Visit my portfolio
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)
