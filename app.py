
import re
import time
import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PREMIUM VIDYASHILP UNIVERSITY UI
# ============================================================
BASE = Path(__file__).resolve().parent
st.set_page_config(
    page_title="Vidyashilp AI Academic Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Locate logo automatically if uploaded beside the app.
LOGO_CANDIDATES = [
    BASE / "vidyashilp_logo.png",
    BASE / "logo.png",
    BASE / "c4994386-5ee2-4699-899b-6a7bc2b5376c.png",
]
LOGO = next((p for p in LOGO_CANDIDATES if p.exists()), None)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

html, body, [class*="css"] { font-family: Inter, sans-serif; }
.block-container { max-width: 1180px; padding: 0.8rem 2rem 2rem; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }

/* Header */
.vu-header {
    display:flex; align-items:center; justify-content:space-between;
    padding: 14px 4px 12px; border-bottom:1px solid #e5e7eb; margin-bottom:18px;
}
.vu-brand { display:flex; align-items:center; gap:13px; }
.vu-logo {
    width:54px; height:54px; object-fit:contain; border-radius:12px;
}
.vu-name { font-family:Manrope,sans-serif; font-size:19px; font-weight:800; color:#123b86; line-height:1.05; }
.vu-sub { font-size:11px; color:#64748b; margin-top:4px; }
.vu-status {
    display:inline-flex; align-items:center; gap:7px; padding:7px 12px;
    border:1px solid #bbf7d0; background:#f0fdf4; color:#166534;
    border-radius:999px; font-size:12px; font-weight:600;
}
.dot { width:7px; height:7px; background:#22c55e; border-radius:50%; }

/* Hero */
.hero {
    border-radius:24px; padding:28px 30px; color:white;
    background:linear-gradient(115deg,#0b2a63 0%,#123f91 52%,#294ea3 100%);
    box-shadow:0 14px 40px rgba(15,42,99,.18); position:relative; overflow:hidden;
}
.hero:after {
    content:""; position:absolute; width:240px; height:240px; border-radius:50%;
    right:-80px; top:-100px; background:rgba(255,255,255,.08);
}
.hero h1 { font-family:Manrope,sans-serif; font-size:34px; margin:0 0 8px; }
.hero p { margin:0; opacity:.88; max-width:650px; font-size:14px; }
.badges { display:flex; gap:8px; margin-top:17px; flex-wrap:wrap; }
.badge { padding:6px 10px; border-radius:999px; background:rgba(255,255,255,.13); font-size:11px; }

/* Robot */
.robot-wrap { display:flex; align-items:center; gap:13px; margin:8px 0 4px; }
.robot {
    width:58px; height:58px; border-radius:19px; position:relative;
    background:linear-gradient(145deg,#f8fafc,#dbeafe); border:2px solid #93c5fd;
    box-shadow:0 8px 22px rgba(59,130,246,.20);
}
.robot:before {
    content:""; position:absolute; left:11px; top:13px; width:32px; height:27px;
    border-radius:12px; background:#ffffff; border:1px solid #bfdbfe;
}
.eye1,.eye2 { position:absolute; top:24px; width:6px; height:6px; border-radius:50%; background:#2563eb; z-index:2; }
.eye1 {left:19px}.eye2 {left:33px}
.robot:after {
    content:""; position:absolute; left:26px; top:5px; width:5px; height:9px;
    border-radius:5px; background:#2563eb;
}
.robot-name { font-family:Manrope,sans-serif; font-weight:800; color:#0f2f67; font-size:15px; }
.robot-msg { color:#64748b; font-size:12px; margin-top:2px; }

/* Chat */
.chat-card {
    margin-top:18px; border:1px solid #e2e8f0; border-radius:22px;
    background:white; box-shadow:0 10px 35px rgba(15,23,42,.06); padding:20px;
}
.chat-title { font-family:Manrope,sans-serif; font-size:22px; font-weight:800; color:#0f172a; }
.chat-sub { color:#64748b; font-size:13px; margin:3px 0 15px; }
.example {
    border:1px solid #e5e7eb; border-radius:12px; padding:9px 11px;
    font-size:12px; color:#334155; background:#fff; min-height:42px;
}

/* Hide excessive default chrome */
div[data-testid="stMetric"] { border:1px solid #e5e7eb; border-radius:15px; background:white; }
div[data-testid="stChatMessage"] { border-radius:16px; }
.stButton button { border-radius:11px; font-weight:600; }
.stTextInput input { border-radius:12px; }
</style>

<style>
/* Clean interaction controls */
div[data-testid="stChatInput"] {
    margin-top: 12px;
}
div[data-testid="stChatInput"] textarea {
    border-radius: 18px !important;
    min-height: 54px !important;
    font-size: 15px !important;
    background: #ffffff !important;
    border: 1px solid #dce3ee !important;
}
.stButton > button {
    border-radius: 14px !important;
    border: 1px solid #e3e8f1 !important;
    background: #ffffff !important;
    color: #173d82 !important;
    min-height: 44px !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    border-color: #8db8f6 !important;
    background: #f5f8ff !important;
}
</style>
""", unsafe_allow_html=True)

# Data loading
DATA = BASE / "data" if (BASE / "data").exists() else BASE

REQUIRED_FILES = [
    "course_master.csv","semester_offerings.csv","degree_requirements.csv",
    "students.csv","student_course_history.csv","rag_documents.csv",
    "minor_courses.csv","structure_courses.csv","prerequisite_table.csv",
]
missing_files = [f for f in REQUIRED_FILES if not (DATA / f).exists()]
if missing_files:
    st.error("University data files missing: " + ", ".join(missing_files))
    st.stop()

@st.cache_data
def load_data(data_dir):
    return {
        "course_master": pd.read_csv(data_dir / "course_master.csv"),
        "semester_offerings": pd.read_csv(data_dir / "semester_offerings.csv"),
        "degree_requirements": pd.read_csv(data_dir / "degree_requirements.csv"),
        "students": pd.read_csv(data_dir / "students.csv"),
        "history": pd.read_csv(data_dir / "student_course_history.csv"),
        "rag_documents": pd.read_csv(data_dir / "rag_documents.csv"),
        "minor_courses": pd.read_csv(data_dir / "minor_courses.csv"),
        "structure_courses": pd.read_csv(data_dir / "structure_courses.csv"),
        "prerequisite_table": pd.read_csv(data_dir / "prerequisite_table.csv"),
    }

D = load_data(DATA)
course_master = D["course_master"]; semester_offerings = D["semester_offerings"]
degree_requirements = D["degree_requirements"]; students = D["students"]
history = D["history"]; rag_documents = D["rag_documents"]
minor_courses = D["minor_courses"]; structure_courses = D["structure_courses"]
prerequisite_table = D["prerequisite_table"]

@st.cache_resource
def build_retriever(texts):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=True, sublinear_tf=True, min_df=1)
    matrix = vectorizer.fit_transform(texts.astype(str))
    return vectorizer, matrix

vectorizer, rag_matrix = build_retriever(rag_documents["text"])

def norm_text(value):
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def norm_code(value):
    return re.sub(r"\s+", "", str(value).strip().upper())

def retrieve_rag(query, top_k=6, threshold=0.08):
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, rag_matrix)[0]
    order = scores.argsort()[::-1]
    rows=[]
    for idx in order:
        score=float(scores[idx])
        if score < threshold: continue
        row=rag_documents.iloc[idx].copy()
        row["retrieval_score"]=score
        rows.append(row)
        if len(rows)>=top_k: break
    return pd.DataFrame(rows)

def course_by_code(code):
    code=norm_code(code)
    x=course_master[course_master["course_code"].astype(str).map(norm_code)==code]
    return None if x.empty else x.iloc[0].to_dict()

def courses_by_name(question):
    q=norm_text(question); matches=[]
    for _,row in course_master.iterrows():
        title=str(row["course_title"]).strip(); tn=norm_text(title)
        if tn and tn in q:
            matches.append({"course_code":norm_code(row["course_code"]),"course_title":title})
    if not matches:
        for _,row in course_master.iterrows():
            tn=norm_text(row["course_title"])
            if tn and (q==tn or tn.startswith(q) or q.startswith(tn)):
                matches.append({"course_code":norm_code(row["course_code"]),"course_title":str(row["course_title"]).strip()})
    return list({x["course_code"]:x for x in matches}.values())

def student_row(student_id):
    if not student_id or student_id=="New / General User": return None
    x=students[students["student_id"].astype(str).str.upper()==str(student_id).upper()]
    return None if x.empty else x.iloc[0]

def student_history_rows(student_id):
    if not student_id or student_id=="New / General User": return history.iloc[0:0].copy()
    return history[history["student_id"].astype(str).str.upper()==str(student_id).upper()].copy()

def source_for_course(code):
    code=norm_code(code)
    x=semester_offerings[semester_offerings["course_code"].astype(str).map(norm_code)==code]
    if x.empty: x=prerequisite_table[prerequisite_table["course_code"].astype(str).map(norm_code)==code]
    if x.empty: return "No directly matched structured source row was found."
    r=x.iloc[0]
    return f"{r.get('source_sheet','unavailable')}, row {r.get('source_row','unavailable')}"

def course_records(code):
    return semester_offerings[semester_offerings["course_code"].astype(str).map(norm_code)==norm_code(code)].copy()

def is_nil_prerequisite(value):
    return str(value).strip().upper() in {"","NIL","NONE","NAN"}

def prerequisite_codes(value):
    return [norm_code(x) for x in re.findall(r"\b[A-Z]{3,6}\s*\d{3,4}\b",str(value).upper())]

def find_course_from_question(question):
    codes=re.findall(r"\b[A-Za-z]{3,6}\s*\d{3,4}\b",question)
    if codes:
        code=norm_code(codes[0]); course=course_by_code(code)
        return {"type":"identified","course":{"course_code":code,"course_title":str(course["course_title"])}} if course else {"type":"not_found","course":None}
    matches=courses_by_name(question)
    if len(matches)==1: return {"type":"identified","course":matches[0]}
    if len(matches)>1: return {"type":"ambiguous","course":matches}
    return {"type":"none","course":None}

# The grounded decision function is intentionally conservative.
INSUFFICIENT="I do not have enough information to determine this reliably from the provided university data."


# ============================================================
# BROAD GROUNDED DATABASE QA ENGINE
# ============================================================
# Design principle:
#   Natural-language question
#        -> retrieve university records
#        -> identify relevant entity/intent
#        -> answer from structured fields
#        -> if evidence is insufficient/conflicting, do NOT guess
#
# This deliberately does not let a free-form model invent university rules.

INSUFFICIENT = (
    "I do not have enough information to determine this reliably "
    "from the provided university data."
)

def _clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()

def _norm(v):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", _clean(v).lower())).strip()

def _code(v):
    return re.sub(r"\s+", "", _clean(v).upper())

def _extract_codes(q):
    return [_code(x) for x in re.findall(r"\b[A-Za-z]{2,8}\s*\d{2,5}\b", q)]

def _course_matches(q):
    qn = _norm(q)
    matches = []

    # Explicit course code
    for c in _extract_codes(q):
        row = course_master[course_master["course_code"].astype(str).map(_code) == c]
        if not row.empty:
            matches.append({
                "course_code": c,
                "course_title": _clean(row.iloc[0]["course_title"])
            })

    # Exact / contained course title
    for _, row in course_master.iterrows():
        title = _clean(row["course_title"])
        tn = _norm(title)
        if tn and (tn in qn or qn == tn):
            matches.append({
                "course_code": _code(row["course_code"]),
                "course_title": title
            })

    # Fuzzy-ish token overlap for natural questions.
    q_tokens = set(qn.split())
    if len(q_tokens) >= 2:
        for _, row in course_master.iterrows():
            title = _clean(row["course_title"])
            tokens = set(_norm(title).split())
            overlap = len(tokens & q_tokens)
            if overlap >= max(2, min(3, len(tokens))):
                matches.append({
                    "course_code": _code(row["course_code"]),
                    "course_title": title
                })

    unique = {}
    for m in matches:
        unique[m["course_code"]] = m
    return list(unique.values())

def _student_matches(q):
    q_upper = q.upper()
    ids = [sid for sid in students["student_id"].astype(str) if sid.upper() in q_upper]
    return ids

def _structure_matches(q):
    q_upper = q.upper()
    return [
        s for s in degree_requirements["academic_structure"].dropna().astype(str).unique()
        if s.upper() in q_upper
    ]

def _semester_matches(q):
    found = re.findall(r"\b(?:semester|sem|s)\s*[-:]?\s*(\d{1,2})\b", q, re.I)
    return [f"S{x}" for x in found]

def _is_question_about(q, *terms):
    qn = _norm(q)
    return any(t in qn for t in terms)

def _fmt_num(v):
    try:
        x = float(v)
        return str(int(x)) if x.is_integer() else f"{x:g}"
    except Exception:
        return _clean(v)

def _course_source(code):
    x = semester_offerings[
        semester_offerings["course_code"].astype(str).map(_code) == _code(code)
    ]
    if not x.empty:
        r = x.iloc[0]
        return f"{_clean(r.get('source_sheet','unavailable'))}, row {_clean(r.get('source_row','unavailable'))}"
    x = prerequisite_table[
        prerequisite_table["course_code"].astype(str).map(_code) == _code(code)
    ]
    if not x.empty:
        r = x.iloc[0]
        return f"prerequisite table, row {_clean(r.get('source_row','unavailable'))}"
    return ""

def _course_answer(q, course):
    code = course["course_code"]
    title = course["course_title"]
    master = course_master[
        course_master["course_code"].astype(str).map(_code) == code
    ]
    if master.empty:
        return None

    m = master.iloc[0]
    offerings = semester_offerings[
        semester_offerings["course_code"].astype(str).map(_code) == code
    ]
    prereq_rows = prerequisite_table[
        prerequisite_table["course_code"].astype(str).map(_code) == code
    ]

    qn = _norm(q)
    parts = []

    # Prerequisite
    if "prerequisite" in qn or "pre requisite" in qn or "pre-requisite" in qn:
        vals = []
        for v in [m.get("prerequisite", "")] + offerings.get("prerequisite", pd.Series(dtype=str)).tolist():
            if _clean(v) and _norm(v) not in {"nil", "none", "nan"}:
                vals.append(_norm(v))
        if len(set(vals)) > 1:
            return (
                f"The provided university records contain conflicting prerequisite information "
                f"for {title} ({code}). {INSUFFICIENT}"
            )
        return f"The documented prerequisite for **{title} ({code})** is **{_clean(m.get('prerequisite','unavailable'))}**."

    # Credits
    if "credit" in qn:
        vals = sorted(set(
            pd.to_numeric(offerings["credits"], errors="coerce").dropna().tolist()
        ))
        if not vals:
            return None
        if len(vals) == 1:
            return f"**{title} ({code})** carries **{_fmt_num(vals[0])} credit(s)**."
        return f"**{title} ({code})** has these credit values in the provided offering records: **{', '.join(_fmt_num(x) for x in vals)}**."

    # Semester / offering
    if any(x in qn for x in ["offered", "offering", "available", "taught"]) or "semester" in qn:
        sems = _semester_matches(q)
        if sems:
            x = offerings[offerings["semester"].astype(str).str.upper().isin(sems)]
            if x.empty:
                return f"**{title} ({code})** is not listed in the provided offering data for **{', '.join(sems)}**."
            structures = sorted(x["academic_structure"].dropna().astype(str).unique())
            return f"**{title} ({code})** is listed for **{', '.join(sems)}** in the provided offering data" + (
                f" under {', '.join(structures)}." if structures else "."
            )

        if not offerings.empty:
            sems_all = sorted(offerings["semester"].dropna().astype(str).unique())
            return f"**{title} ({code})** is listed in these semester offering records: **{', '.join(sems_all)}**."

    # Minor
    if "minor" in qn:
        rows = minor_courses[
            minor_courses["course_code"].astype(str).map(_code) == code
        ]
        if not rows.empty:
            minors = sorted(rows["minor"].dropna().astype(str).unique())
            return f"**{title} ({code})** appears under: **{', '.join(minors)}**."

    # Structure / category
    if any(x in qn for x in ["structure", "program core", "programme core", "category"]):
        rows = structure_courses[
            structure_courses["course_number"].astype(str).map(_code) == code
        ]
        if not rows.empty:
            cats = sorted(rows["structure_category"].dropna().astype(str).unique())
            return f"**{title} ({code})** is classified under: **{', '.join(cats)}**."

    # Generic course information: answer from the database, not a canned model answer.
    if any(x in qn for x in [
        "what is", "tell me", "information", "details", "about",
        "describe", "which", "show", "give me"
    ]):
        bits = [f"**{title} ({code})**"]
        if _clean(m.get("prerequisite", "")):
            bits.append(f"Prerequisite: **{_clean(m['prerequisite'])}**")
        credits = sorted(set(pd.to_numeric(offerings["credits"], errors="coerce").dropna().tolist()))
        if credits:
            bits.append(f"Credits: **{', '.join(_fmt_num(x) for x in credits)}**")
        sems = sorted(offerings["semester"].dropna().astype(str).unique())
        if sems:
            bits.append(f"Offerings: **{', '.join(sems)}**")
        return "\n\n".join(bits)

    return None

def _list_query_answer(q):
    qn = _norm(q)
    sems = _semester_matches(q)
    structures = _structure_matches(q)

    # Courses offered in a semester
    if sems and any(x in qn for x in ["courses", "subjects", "offerings", "offered"]):
        x = semester_offerings[
            semester_offerings["semester"].astype(str).str.upper().isin(sems)
        ]
        if structures:
            x = x[x["academic_structure"].astype(str).str.upper().isin([s.upper() for s in structures])]
        if x.empty:
            return f"No course offering records were found for **{', '.join(sems)}**."
        x = x.drop_duplicates(subset=["course_code"])
        names = [
            f"{_clean(r.course_title)} ({_code(r.course_code)})"
            for r in x.itertuples()
        ]
        # Keep the main answer readable.
        return f"Courses listed for **{', '.join(sems)}**: " + "; ".join(names)

    # Minor listing
    if "minor" in qn and any(x in qn for x in ["courses", "subjects", "list", "which"]):
        minor_terms = [
            _clean(x) for x in minor_courses["minor"].dropna().astype(str).unique()
            if _norm(x) in qn or _norm(x).replace("minor","").strip() in qn
        ]
        x = minor_courses
        if minor_terms:
            x = x[x["minor"].astype(str).isin(minor_terms)]
        else:
            # Token overlap against minor names.
            q_tokens = set(qn.split())
            scored = []
            for m in minor_courses["minor"].dropna().astype(str).unique():
                score=len(set(_norm(m).split()) & q_tokens)
                scored.append((score,m))
            best=max(scored, default=(0,""))
            if best[0] > 0:
                x=x[x["minor"].astype(str)==best[1]]
        if not x.empty:
            x=x.drop_duplicates(subset=["course_code"])
            return "Courses in the requested minor: " + "; ".join(
                f"{_clean(r.course_title)} ({_code(r.course_code)})"
                for r in x.itertuples()
            )

    # Degree requirements
    if any(x in qn for x in ["degree requirement", "degree requirements", "required credits", "requirements"]):
        if structures:
            x=degree_requirements[
                degree_requirements["academic_structure"].astype(str).str.upper().isin([s.upper() for s in structures])
            ]
            if not x.empty:
                return "\n\n".join(
                    f"**{_clean(r.component)}:** {_fmt_num(r.required_credits)} credits"
                    for r in x.itertuples()
                )
        return (
            "The database contains multiple academic structures. "
            "Please specify the academic structure (for example, Struct_2024) "
            "so I can return the correct requirement."
        )

    return None

def _student_answer(q, student_id):
    if not student_id or student_id == "New / General User":
        return None
    s = student_row(student_id)
    h = student_history_rows(student_id)
    if s is None:
        return None
    qn = _norm(q)

    if any(x in qn for x in ["profile", "my details", "my information", "student details", "my academic"]):
        return (
            f"Student **{_clean(s.student_id)}** — {_clean(s.programme)}, "
            f"batch {_clean(s.batch)}, semester {_clean(s.current_semester)}, "
            f"{_clean(s.total_credits)} total credits, minor: {_clean(s['minor'])}."
        )

    codes = _extract_codes(q)
    course_matches = _course_matches(q)
    code = codes[0] if codes else (course_matches[0]["course_code"] if len(course_matches)==1 else None)

    if code:
        hh = h[h["course_code"].astype(str).map(_code)==code]
        if any(x in qn for x in ["grade", "marks", "score", "pass", "passed", "fail", "failed", "status", "result"]):
            if hh.empty:
                return f"I do not have a course-history record for **{code}** for student **{student_id}**."
            r=hh.iloc[0]
            return f"For **{code}**, the student record shows **{_clean(r.status)}** with grade **{_clean(r.grade)}**."

    # Broad student-history query
    if any(x in qn for x in ["my courses", "courses i took", "course history", "academic history", "my results"]):
        if h.empty:
            return f"I do not have course-history records for **{student_id}**."
        return "; ".join(
            f"{_code(r.course_code)}: {_clean(r.status)} ({_clean(r.grade)})"
            for r in h.itertuples()
        )

    return None

def _generic_rag_answer(q):
    retrieved = retrieve_rag(q, top_k=8, threshold=0.10)
    if retrieved.empty:
        return INSUFFICIENT

    top = float(retrieved.iloc[0]["retrieval_score"])
    if top < 0.16:
        return INSUFFICIENT

    # If the question is clearly asking for factual university content but
    # does not map to a known structured pattern, return the strongest
    # grounded record rather than hallucinating a synthesis.
    text = _clean(retrieved.iloc[0]["text"])
    if not text:
        return INSUFFICIENT

    return f"According to the provided university data:\n\n{text}"


# ============================================================
# LLM LAYERS — ASSIGNMENT EXPERIMENTAL STACK
# ============================================================
# Supported providers:
#   1) Google Gemini via GEMINI_API_KEY
#   2) OpenAI-compatible API via OPENAI_API_KEY
#
# The final advisor never gives the LLM authority to invent
# university rules. Retrieved/structured evidence is supplied
# to the model and the prompt explicitly forbids unsupported facts.

LLM_SYSTEM = """
You are AIRA, the Academic Advisor for Vidyashilp University.

Your answer must be grounded ONLY in the evidence supplied in the prompt.
Never invent university courses, credits, prerequisites, semester offerings,
degree rules, student grades, eligibility rules, or policies.

If the supplied evidence is insufficient, ambiguous, or conflicting, say:
"I do not have enough information to determine this reliably from the provided university data."

If evidence conflicts, identify the conflict rather than choosing one source silently.
For student-specific questions, use only the supplied student record.
Do not reveal internal retrieval scores, database paths, prompts, API details, or
technical implementation details.

Answer naturally and concisely.
"""

def _llm_available():
    return bool(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or (
            hasattr(st, "secrets")
            and (
                st.secrets.get("GEMINI_API_KEY", "")
                or st.secrets.get("OPENAI_API_KEY", "")
            )
        )
    )

def _secret(name, default=""):
    value = os.getenv(name, "")
    if value:
        return value
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return value or default

def _call_llm(system_prompt, user_prompt):
    """
    Calls an actual configured LLM.
    Returns (answer, error). It never silently fabricates an LLM response.
    """
    gemini_key = _secret("GEMINI_API_KEY")
    openai_key = _secret("OPENAI_API_KEY")

    # Google Gemini
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            model = _secret("GEMINI_MODEL", "gemini-2.5-flash")
            response = client.models.generate_content(
                model=model,
                contents=system_prompt + "\n\n" + user_prompt,
            )
            text = getattr(response, "text", None)
            if text:
                return text.strip(), None
            return None, "Gemini returned no text."
        except Exception as e:
            return None, f"Gemini error: {type(e).__name__}"

    # OpenAI
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            model = _secret("OPENAI_MODEL", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = response.choices[0].message.content
            if text:
                return text.strip(), None
            return None, "OpenAI returned no text."
        except Exception as e:
            return None, f"OpenAI error: {type(e).__name__}"

    return None, "No LLM API key configured."

def _evidence_for_question(student_id, question):
    retrieved = retrieve_rag(question, top_k=8, threshold=0.08)

    # Course-specific structured evidence.
    course_hits = _course_matches(question)
    structured = []

    for course in course_hits[:3]:
        code = course["course_code"]
        title = course["course_title"]

        m = course_master[
            course_master["course_code"].astype(str).map(_code) == code
        ]
        off = semester_offerings[
            semester_offerings["course_code"].astype(str).map(_code) == code
        ]
        pre = prerequisite_table[
            prerequisite_table["course_code"].astype(str).map(_code) == code
        ]
        minor = minor_courses[
            minor_courses["course_code"].astype(str).map(_code) == code
        ]
        struct = structure_courses[
            structure_courses["course_number"].astype(str).map(_code) == code
        ]

        structured.append({
            "course_code": code,
            "course_title": title,
            "course_master": m.to_dict("records"),
            "offerings": off.to_dict("records"),
            "prerequisites": pre.to_dict("records"),
            "minor_records": minor.to_dict("records"),
            "structure_records": struct.to_dict("records"),
        })

    student_data = None
    if student_id and student_id != "New / General User":
        s = student_row(student_id)
        h = student_history_rows(student_id)
        if s is not None:
            student_data = {
                "profile": s.to_dict(),
                "course_history": h.to_dict("records"),
            }

    rag = []
    if not retrieved.empty:
        for _, row in retrieved.iterrows():
            rag.append({
                "source_type": str(row.get("source_type", "")),
                "source_row": str(row.get("source_row", "")),
                "text": str(row.get("text", "")),
            })

    return {
        "structured_course_evidence": structured,
        "student_evidence": student_data,
        "rag_evidence": rag,
    }

def _llm_answer(question, evidence, layer):
    if layer == "basic":
        user_prompt = f"""
Student question:
{question}

Answer the question using your general language-model knowledge.
Do not claim a Vidyashilp-specific rule unless it is in the supplied evidence.
"""
        return _call_llm(LLM_SYSTEM, user_prompt)

    if layer == "structured":
        user_prompt = f"""
Student question:
{question}

Follow these rules:
1. Identify exactly what the user is asking.
2. State only information you can support.
3. If information is missing, say so.
4. Do not invent a university rule.

Use this structured response format internally:
- Answer
- Evidence basis
- Uncertainty, if any

Do not expose the headings unless useful to the student.
"""
        return _call_llm(LLM_SYSTEM, user_prompt)

    evidence_json = json.dumps(evidence, ensure_ascii=False, default=str)

    if layer == "rag":
        user_prompt = f"""
Student question:
{question}

Retrieved university evidence:
{evidence_json}

Answer ONLY from the retrieved university evidence.
If it does not answer the question, use the required insufficient-information
sentence. Never fill gaps using general knowledge.
"""
        return _call_llm(LLM_SYSTEM, user_prompt)

    # Final layer: RAG + structured student data.
    user_prompt = f"""
Student question:
{question}

Verified university evidence:
{evidence_json}

This is the final grounded advisor layer.
Use the structured university records and, when present, the student's structured
record. Do not infer facts that are not present.

For eligibility or student-history questions, distinguish:
- documented facts
- missing information
- conflicting information

If a registration/eligibility policy itself is not documented, do NOT invent one.
Say that the available prerequisite/history evidence is insufficient to establish
the registration policy.

Return the most direct useful answer supported by the evidence.
"""
    return _call_llm(LLM_SYSTEM, user_prompt)

def _llm_layer_answer(student_id, question, state):
    """
    Runs the selected experimental layer. The final layer uses RAG + student data.
    If an LLM is not configured, the app tells the user exactly what is missing
    rather than pretending an LLM was used.
    """
    layer = st.session_state.get("llm_layer", "final")

    if not _llm_available():
        return (
            "The LLM layer is not configured yet. Add GEMINI_API_KEY or "
            "OPENAI_API_KEY in Streamlit Secrets to run this LLM layer.",
            state,
            retrieve_rag(question),
        )

    evidence = _evidence_for_question(student_id, question)

    if layer == "basic":
        answer, error = _llm_answer(question, evidence, "basic")
    elif layer == "structured":
        answer, error = _llm_answer(question, evidence, "structured")
    elif layer == "rag":
        answer, error = _llm_answer(question, evidence, "rag")
    else:
        answer, error = _llm_answer(question, evidence, "final")

    if error:
        return (
            "I could not obtain a reliable LLM answer right now. "
            "Please check the configured LLM connection.",
            state,
            retrieve_rag(question),
        )

    return answer, state, retrieve_rag(question)

def _deterministic_answer_question(student_id, question, state):
    q = str(question).strip()
    if not q:
        return "Please type a question.", state, pd.DataFrame()

    # Follow-up for ambiguous course names.
    if state.get("awaiting_course_code"):
        if re.fullmatch(r"[A-Za-z]{2,8}\s*\d{2,5}", q):
            code = _code(q)
            if course_by_code(code) is None:
                return f"Course code **{code}** is not present in the provided university data.", state, pd.DataFrame()
            original = state.get("original_question", "")
            state.clear()
            return answer_question(student_id, original + " " + code, state)
        return "Please provide the course code so I can identify the correct course.", state, pd.DataFrame()

    # Student-specific questions first.
    student_result = _student_answer(q, student_id)
    if student_result:
        return student_result, state, retrieve_rag(q)

    # Course ambiguity / identification.
    courses = _course_matches(q)
    if len(courses) > 1 and not _extract_codes(q):
        state["awaiting_course_code"] = True
        state["original_question"] = q
        return (
            "I found more than one course matching that name. "
            "Please enter the course code: " +
            ", ".join(f"{x['course_code']} — {x['course_title']}" for x in courses)
        ), state, retrieve_rag(q)

    if len(courses) == 1:
        result = _course_answer(q, courses[0])
        if result:
            return result, state, retrieve_rag(q)

    # General list / requirements queries.
    result = _list_query_answer(q)
    if result:
        return result, state, retrieve_rag(q)

    # If the question appears student-specific but no profile exists.
    if any(x in _norm(q) for x in ["my ", "did i", "have i", "can i", "am i", "myself"]):
        return (
            "Please select a student profile so I can answer from the student database. "
            "I will not assume your academic record."
        ), state, retrieve_rag(q)

    # Broad RAG fallback.
    return _generic_rag_answer(q), state, retrieve_rag(q)


def answer_question(student_id, question, state):
    q = str(question).strip()
    if not q:
        return "Please type a question.", state, pd.DataFrame()

    # The four assignment layers are available.
    # The deployed student-facing default is the final grounded layer.
    layer = st.session_state.get("llm_layer", "final")

    # For actual LLM layers, use the LLM.
    if layer in {"basic", "structured", "rag", "final"}:
        return _llm_layer_answer(student_id, q, state)

    # Deterministic verified database engine retained as an internal safety mode.
    return _deterministic_answer_question(student_id, q, state)



# ---------- LLM LAYER ----------
# Kept compact so the student-facing screen stays clean.
if "llm_layer" not in st.session_state:
    st.session_state.llm_layer = "final"

with st.expander("Advisor mode", expanded=False):
    layer_labels = {
        "basic": "Basic LLM",
        "structured": "Structured Prompting",
        "rag": "LLM + RAG",
        "final": "LLM + RAG + Structured Student Data",
    }
    current = st.session_state.llm_layer
    selected_label = st.selectbox(
        "Layer",
        list(layer_labels.values()),
        index=list(layer_labels.keys()).index(current),
    )
    st.session_state.llm_layer = next(
        k for k, v in layer_labels.items() if v == selected_label
    )

# ---------- PROFILE SELECTION ----------
profile_options = ["New / General User"] + students["student_id"].astype(str).tolist()

if "selected_student" not in st.session_state:
    st.session_state.selected_student = "New / General User"

profile_col1, profile_col2, profile_col3 = st.columns([1, 1.4, 1])
with profile_col2:
    selected_student = st.selectbox(
        "Profile",
        profile_options,
        index=profile_options.index(st.session_state.selected_student),
        label_visibility="collapsed",
    )
st.session_state.selected_student = selected_student


# ============================================================
# AIRA — PREMIUM INTERACTIVE STUDENT EXPERIENCE
# ============================================================

import base64
from html import escape

# ---------- BRAND ASSET ----------
logo_html = ""
if LOGO and LOGO.exists():
    b64 = base64.b64encode(LOGO.read_bytes()).decode("utf-8")
    logo_html = f'<img class="aira-logo" src="data:image/png;base64,{b64}" alt="Vidyashilp University">'
else:
    logo_html = '<div class="aira-logo-fallback">V</div>'

# ---------- SESSION ----------
profile_options = ["New / General User"] + students["student_id"].astype(str).tolist()

if "selected_student" not in st.session_state:
    st.session_state.selected_student = "New / General User"
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "advisor_state" not in st.session_state:
    st.session_state.advisor_state = {}
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------- PREMIUM VISUAL SYSTEM ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}
.block-container {
    max-width: 1180px;
    padding: 1rem 1.25rem 2rem;
}
header[data-testid="stHeader"] { background: transparent; }

/* Remove Streamlit clutter */
div[data-testid="stMetric"],
div[data-testid="stCaptionContainer"] { display:none !important; }
[data-testid="stToolbar"] { opacity:.55; }

/* ---------- HEADER ---------- */
.aira-header {
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:4px 4px 14px;
    border-bottom:1px solid #edf1f7;
    margin-bottom:16px;
}
.aira-brand {
    display:flex;
    align-items:center;
    gap:12px;
}
.aira-logo {
    width:48px;height:48px;object-fit:contain;
}
.aira-logo-fallback {
    width:48px;height:48px;border-radius:15px;
    background:#123f91;color:#fff;
    display:flex;align-items:center;justify-content:center;
    font-weight:800;font-size:24px;
}
.aira-brand-name {
    color:#123f91;
    font-family:Manrope,sans-serif;
    font-weight:800;
    font-size:17px;
    letter-spacing:-.4px;
}
.aira-brand-sub {
    color:#8a96a8;font-size:10px;margin-top:2px;
}
.aira-status {
    display:flex;align-items:center;gap:7px;
    padding:8px 12px;border-radius:999px;
    color:#16733b;background:#f3fff7;border:1px solid #c7efd4;
    font-size:11px;font-weight:600;
}
.status-dot {
    width:7px;height:7px;border-radius:50%;
    background:#22c55e;box-shadow:0 0 0 4px #dcfce7;
}

/* ---------- PROFILE ---------- */
.profile-wrap {
    display:flex;justify-content:flex-end;margin-bottom:10px;
}
div[data-testid="stSelectbox"] {
    max-width:340px;
}
div[data-testid="stSelectbox"] label { display:none; }
div[data-testid="stSelectbox"] > div > div {
    border-radius:13px !important;
}

/* ---------- HERO ---------- */
.aira-hero {
    position:relative;
    overflow:hidden;
    min-height:405px;
    border-radius:30px;
    background:
      radial-gradient(circle at 50% 42%, rgba(96,165,250,.25), transparent 17%),
      radial-gradient(circle at 50% 48%, rgba(59,130,246,.16), transparent 37%),
      linear-gradient(145deg,#071b42,#0e347d 52%,#1a469d);
    box-shadow:0 25px 65px rgba(18,59,134,.18);
    display:flex;
    align-items:center;
    justify-content:center;
}
.aira-ring {
    position:absolute;border:1px solid rgba(191,219,254,.20);
    border-radius:50%;pointer-events:none;
}
.r1{width:300px;height:300px}.r2{width:430px;height:430px}.r3{width:590px;height:590px;opacity:.55}
.star {
    position:absolute;width:6px;height:6px;border-radius:50%;
    background:#93c5fd;box-shadow:0 0 13px rgba(147,197,253,.9);
    animation:airaFloat 5s ease-in-out infinite;
}
.s1{left:19%;top:21%}.s2{right:20%;top:28%;animation-delay:1s}
.s3{left:26%;bottom:20%;animation-delay:2s}.s4{right:25%;bottom:18%;animation-delay:3s}
@keyframes airaFloat {50%{transform:translateY(-12px);opacity:.45}}

.aira-bot {
    position:relative;width:190px;height:235px;
    animation:airaBreath 4s ease-in-out infinite;
    z-index:2;
}
@keyframes airaBreath {50%{transform:translateY(-5px)}}

/* Hair */
.bot-hair {
    position:absolute;left:24px;top:5px;width:142px;height:92px;
    border-radius:75px 75px 45px 45px;
    background:linear-gradient(145deg,#f4f8ff,#9fc9ff);
    box-shadow:inset -10px -8px rgba(37,99,235,.12);
}
.bot-hair:after {
    content:"";position:absolute;right:-3px;top:36px;
    width:30px;height:65px;border-radius:0 25px 25px 25px;
    background:#b9d6ff;
}

/* Head */
.bot-head {
    position:absolute;left:35px;top:38px;width:120px;height:108px;
    border-radius:43px 43px 39px 39px;
    background:linear-gradient(160deg,#fff,#e8f2ff);
    border:3px solid #91bdf7;
    box-shadow:0 18px 35px rgba(0,0,0,.22);
}
.bot-ear {
    position:absolute;top:76px;width:15px;height:31px;border-radius:10px;
    background:#bfdbfe;border:2px solid #8ebaf2;
}
.bot-ear.l{left:21px}.bot-ear.r{right:21px}
.bot-eye {
    position:absolute;top:73px;width:14px;height:14px;border-radius:50%;
    background:#2563eb;box-shadow:0 0 14px rgba(37,99,235,.65);
}
.bot-eye.l{left:58px}.bot-eye.r{left:108px}
.bot-mouth {
    position:absolute;left:78px;top:102px;width:28px;height:11px;
    border-bottom:3px solid #2563eb;border-radius:0 0 20px 20px;
}
.bot-neck {
    position:absolute;left:82px;top:140px;width:26px;height:20px;
    border-radius:7px;background:#b8d5fa;
}
.bot-body {
    position:absolute;left:52px;top:153px;width:86px;height:65px;
    border-radius:25px 25px 18px 18px;
    background:linear-gradient(160deg,#e4efff,#b8d5fb);
    border:2px solid #8db8f2;
    box-shadow:0 15px 30px rgba(0,0,0,.18);
}
.bot-core {
    position:absolute;left:79px;top:173px;width:32px;height:32px;
    border-radius:11px;background:#fff;border:1px solid #8fbaf2;
    display:flex;align-items:center;justify-content:center;
    color:#17458e;font-size:10px;font-weight:800;
}
.aira-name {
    position:absolute;bottom:30px;color:white;
    font:800 21px Manrope,sans-serif;letter-spacing:-.5px;z-index:3;
}
.aira-role {
    position:absolute;bottom:12px;color:rgba(255,255,255,.68);
    font-size:10px;z-index:3;
}

/* ---------- ANSWER ---------- */
.answer-card {
    max-width:930px;margin:16px auto 12px;
    padding:24px 28px;
    border:1px solid #e3e9f2;border-radius:24px;
    background:#fff;
    box-shadow:0 10px 32px rgba(15,23,42,.06);
}
.answer-kicker {
    color:#2860ae;font-size:10px;font-weight:800;
    letter-spacing:1.4px;text-transform:uppercase;margin-bottom:8px;
}
.answer-body {
    color:#132238;font-size:18px;line-height:1.58;
    font-weight:500;
}
.answer-body strong {color:#123f91;}

/* ---------- ACTION CHIPS ---------- */
.action-title {
    text-align:center;color:#7c8799;font-size:10px;
    letter-spacing:1px;text-transform:uppercase;
    margin:12px 0 7px;
}
.stButton > button {
    border:1px solid #e0e6ef !important;
    border-radius:14px !important;
    background:#fff !important;
    color:#183e80 !important;
    min-height:46px !important;
    font-weight:600 !important;
    transition:all .18s ease !important;
}
.stButton > button:hover {
    border-color:#8bb9f5 !important;
    background:#f5f9ff !important;
    transform:translateY(-1px);
}

/* ---------- INPUT ---------- */
div[data-testid="stChatInput"] {
    margin-top:16px;
}
div[data-testid="stChatInput"] textarea {
    min-height:58px !important;
    border-radius:19px !important;
    border:1px solid #dbe3ef !important;
    font-size:16px !important;
    padding:17px !important;
    box-shadow:0 7px 24px rgba(15,23,42,.06);
}
div[data-testid="stChatInput"] textarea:focus {
    border-color:#7db1f4 !important;
    box-shadow:0 0 0 3px rgba(59,130,246,.10) !important;
}

/* ---------- RESPONSIVE ---------- */
@media (max-width: 700px) {
    .block-container {padding: .6rem .7rem 1.2rem;}
    .aira-hero {min-height:360px;border-radius:24px;}
    .aira-bot {transform:scale(.88);}
    .aira-name {bottom:25px;}
    .aira-role {bottom:8px;}
    .answer-body {font-size:16px;}
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown(f"""
<div class="aira-header">
  <div class="aira-brand">
    {logo_html}
    <div>
      <div class="aira-brand-name">VIDYASHILP UNIVERSITY</div>
      <div class="aira-brand-sub">AIRA · Academic Intelligence & Registration Assistant</div>
    </div>
  </div>
  <div class="aira-status"><span class="status-dot"></span>Advisor online</div>
</div>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="aira-hero">
  <div class="aira-ring r1"></div>
  <div class="aira-ring r2"></div>
  <div class="aira-ring r3"></div>
  <span class="star s1"></span><span class="star s2"></span>
  <span class="star s3"></span><span class="star s4"></span>

  <div class="aira-bot" aria-label="AIRA virtual academic advisor">
    <div class="bot-hair"></div>
    <div class="bot-ear l"></div><div class="bot-ear r"></div>
    <div class="bot-head">
      <div class="bot-eye l"></div><div class="bot-eye r"></div>
      <div class="bot-mouth"></div>
    </div>
    <div class="bot-neck"></div>
    <div class="bot-body"></div>
    <div class="bot-core">AI</div>
  </div>

  <div class="aira-name">AIRA</div>
  <div class="aira-role">Your academic advisor</div>
</div>
""", unsafe_allow_html=True)

# ---------- DIRECT ANSWER ----------
if st.session_state.last_answer:
    safe_answer = escape(str(st.session_state.last_answer)).replace("\n", "<br>")
    # Preserve markdown-style bold from our deterministic answer layer.
    safe_answer = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", safe_answer)
    st.markdown(
        f"""
        <div class="answer-card" aria-live="polite">
          <div class="answer-kicker">AIRA</div>
          <div class="answer-body">{safe_answer}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- VISUAL QUESTION SHORTCUTS ----------
st.markdown('<div class="action-title">Explore</div>', unsafe_allow_html=True)

actions = [
    ("📚  Courses", "Tell me about Leadership and Teamwork Skills"),
    ("◈  Credits", "How many credits does Communication Skills have?"),
    ("◉  Offerings", "Which semester is Leadership and Teamwork Skills offered?"),
    ("♙  My record", "Show my academic details"),
]

cols = st.columns(4)
for col, (label, query) in zip(cols, actions):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.pending_question = query
            st.rerun()

# ---------- INPUT ----------
question = st.chat_input("Ask AIRA anything about your university…")

if st.session_state.pending_question and not question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    answer, new_state, _retrieved = answer_question(
        selected_student,
        question,
        st.session_state.advisor_state,
    )
    st.session_state.advisor_state = new_state
    st.session_state.last_question = question
    st.session_state.last_answer = answer
    st.rerun()

# ---------- FOLLOW-UP ----------
if st.session_state.last_answer:
    f1, f2, f3 = st.columns([1, 1.2, 1])
    with f2:
        if st.button("＋ Ask another question", use_container_width=True):
            st.session_state.last_answer = None
            st.session_state.last_question = None
            st.session_state.advisor_state = {}
            st.rerun()

st.markdown(
    '<div style="text-align:center;color:#a0aabd;font-size:10px;margin-top:12px;">Vidyashilp University · AIRA</div>',
    unsafe_allow_html=True,
)
