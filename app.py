
import re
import time
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

def answer_question(student_id, question, state):
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

# ---------- CLEAN DIRECT-ANSWER EXPERIENCE ----------
# The interface intentionally does NOT display:
# - the user's submitted question as a chat bubble
# - repeated conversation history
# - source rows / retrieval scores
# - technical timing information
# The answer is the only response shown to the student.

if "advisor_state" not in st.session_state:
    st.session_state.advisor_state = {}
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------- ANSWER AREA ----------
if st.session_state.last_answer:
    st.markdown("""
    <style>
    .answer-zone {
        margin: 18px auto 12px;
        max-width: 900px;
        min-height: 92px;
        padding: 22px 28px;
        border-radius: 22px;
        background: rgba(248,250,252,.96);
        border: 1px solid #e5eaf2;
        box-shadow: 0 8px 28px rgba(15,23,42,.05);
    }
    .answer-label {
        color:#64748b;
        font-size:10px;
        text-transform:uppercase;
        letter-spacing:1.2px;
        font-weight:700;
        margin-bottom:7px;
    }
    .answer-text {
        color:#0f172a;
        font-size:18px;
        line-height:1.55;
        font-weight:500;
    }
    .answer-text strong { color:#123f91; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="answer-zone">
          <div class="answer-label">AIRA</div>
          <div class="answer-text">{st.session_state.last_answer.replace(chr(10), '<br>')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- VISUAL QUICK ACTIONS ----------
# These are compact visual actions, not explanatory cards.
actions = [
    ("📚", "Prerequisite", "What is the prerequisite for Leadership and Teamwork Skills?"),
    ("◈", "Credits", "How many credits is Communication Skills?"),
    ("◉", "Offering", "Is Leadership and Teamwork Skills offered in semester 2?"),
    ("♙", "My record", "Show my academic details"),
]

cols = st.columns(4)
for col, (icon, label, query) in zip(cols, actions):
    if col.button(f"{icon}  {label}", use_container_width=True):
        st.session_state.pending_question = query
        st.rerun()

# ---------- INPUT ----------
question = st.chat_input("Ask AIRA anything…")

if st.session_state.pending_question and not question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    answer, new_state, _retrieved = answer_question(
        selected_student,
        question,
        st.session_state.advisor_state
    )
    st.session_state.advisor_state = new_state
    st.session_state.last_question = question
    st.session_state.last_answer = answer
    st.rerun()

# ---------- SMALL RESET ----------
if st.session_state.last_answer:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("＋ Ask another question", use_container_width=True):
            st.session_state.last_answer = None
            st.session_state.last_question = None
            st.session_state.advisor_state = {}
            st.rerun()

st.markdown(
    '<div style="text-align:center;color:#a0aabd;font-size:10px;margin-top:10px;">Vidyashilp University • AIRA</div>',
    unsafe_allow_html=True
)
