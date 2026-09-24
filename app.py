
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

def answer_question(student_id, question, state):
    q=str(question).strip(); ql=q.lower(); retrieved=retrieve_rag(q)

    if state.get("awaiting_course_code"):
        m=re.fullmatch(r"[A-Za-z]{3,6}\s*\d{3,4}",q)
        if not m: return "I found more than one course with that name. Please provide the course code.",state,retrieved
        code=norm_code(m.group()); original=state.get("original_question",""); state.clear()
        if course_by_code(code) is None: return f"Course code {code} is not present in the provided university data.",state,retrieved
        return answer_question(student_id,original+" "+code,state)

    if state.get("awaiting_grade"):
        m=re.search(r"\b(A\+|A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|F|FAIL)\b",q.upper())
        if not m: return "Please provide the grade, such as A, B+, C, D, or F.",state,retrieved
        grade=m.group(1); state.clear()
        return (f"Based on the grade you provided ({grade}), the course status is {'failed' if grade in {'F','FAIL'} else 'passed'}.",state,retrieved)

    if any(k in ql for k in ["graduation credits","credits to graduate","how many credits to graduate","total credits required"]):
        structures=sorted(degree_requirements["academic_structure"].dropna().astype(str).unique())
        m=re.search(r"(Struct_\d{4}(?:_DS)?)",q,re.I)
        if not m:
            return "The database contains multiple academic structures. Please specify your academic structure: "+", ".join(structures)+".",state,retrieved
        structure=m.group(1).upper()
        x=degree_requirements[degree_requirements["academic_structure"].astype(str).str.upper()==structure]
        if x.empty: return f"I could not find {structure} in the provided degree-requirement data.",state,retrieved
        total=x[x["component"].astype(str).str.lower().str.contains("total")]
        if not total.empty: return f"The provided degree-requirement data lists {total.iloc[0]['required_credits']} total credits for {structure}.",state,retrieved
        return f"I found the degree-requirement rows for {structure}, but the data does not provide a single explicit total-credit row.",state,retrieved

    requires_student=any(k in ql for k in ["did i","have i","my grade","my course","can i take","can i register","am i eligible","my eligibility","what should i take","next semester","retake","repeat","again"])
    if requires_student and (student_id=="New / General User" or not student_id):
        return "This is a student-specific question. Select a synthetic student profile so I can use the student database.",state,retrieved

    srow=student_row(student_id); history_df=student_history_rows(student_id)

    if any(k in ql for k in ["my profile","my details","show my details","student details"]):
        if srow is None: return "Please select an existing synthetic student profile first.",state,retrieved
        return f"Student {srow['student_id']} — {srow['programme']}, batch {srow['batch']}, semester {srow['current_semester']}, {srow['total_credits']} total credits, minor: {srow['minor']}.",state,retrieved

    cr=find_course_from_question(q)
    if cr["type"]=="ambiguous":
        state["awaiting_course_code"]=True; state["original_question"]=q
        return "I found multiple courses matching that name. Please provide the course code:\n\n" + "\n".join(f"- **{x['course_code']}** — {x['course_title']}" for x in cr["course"]),state,retrieved
    if cr["type"]=="not_found": return "That course code is not present in the provided university data.",state,retrieved
    if cr["type"]=="none":
        return INSUFFICIENT if retrieved.empty or float(retrieved.iloc[0]["retrieval_score"])<0.16 else "I found related university information, but not enough specific structured evidence for a reliable answer.",state,retrieved

    course=cr["course"]; code=norm_code(course["course_code"]); title=str(course["course_title"]); master=course_by_code(code); offerings=course_records(code)

    if any(k in ql for k in ["prerequisite","pre requisite"]):
        vals=[str(master.get("prerequisite",""))] + offerings["prerequisite"].dropna().astype(str).tolist()
        normalized={norm_text(v) for v in vals if not is_nil_prerequisite(v)}
        if len(normalized)>1:
            return f"The provided sources contain materially different prerequisite information for **{title} ({code})**. I cannot determine the prerequisite reliably without verification.",state,retrieved
        return f"The documented prerequisite for **{title} ({code})** is **{master.get('prerequisite','unavailable')}**.\n\nSource: {source_for_course(code)}",state,retrieved

    if "credit" in ql:
        vals=sorted(set(pd.to_numeric(offerings["credits"],errors="coerce").dropna().tolist()))
        if not vals: return f"I do not have a reliable credit value for {title} ({code}).",state,retrieved
        return f"**{title} ({code})** is listed as **{vals[0]:g} credit(s)**. Source: {source_for_course(code)}.",state,retrieved

    if any(k in ql for k in ["offered","offer","available"]) and "semester" in ql:
        m=re.search(r"(?:semester|sem|s)\s*(\d+)",q,re.I)
        if m:
            target="S"+m.group(1); x=offerings[offerings["semester"].astype(str).str.upper()==target]
            if x.empty: return f"**{title} ({code})** is not listed in the provided semester-offering data for {target}.",state,retrieved
            return f"**{title} ({code})** is listed for **{target}** in the provided offering data.",state,retrieved

    if any(k in ql for k in ["did i fail","did i pass","have i failed","have i passed","my grade"]):
        h=history_df[history_df["course_code"].astype(str).map(norm_code)==code]
        if h.empty: return f"I do not have a course-history record for **{title} ({code})**.",state,retrieved
        return f"Your synthetic student record lists **{title} ({code})** as **{h.iloc[0]['status']}** with grade **{h.iloc[0]['grade']}**.",state,retrieved

    if any(k in ql for k in ["retake","repeat","again"]):
        h=history_df[history_df["course_code"].astype(str).map(norm_code)==code]
        if h.empty: return f"I do not have a course-history record for **{title} ({code})**. The retake policy is not available in the provided data.",state,retrieved
        status=str(h.iloc[0]["status"]).lower()
        if "fail" in status: return f"Your synthetic record shows that you previously failed **{title} ({code})**. The provided university data does not specify the retake policy.",state,retrieved
        return f"Your synthetic record does not show a failed attempt for **{title} ({code})**. The provided university data does not specify the retake policy.",state,retrieved

    if any(k in ql for k in ["can i take","can i register","am i eligible","eligible for"]):
        prereq=str(master.get("prerequisite",""))
        if is_nil_prerequisite(prereq):
            return f"The provided course data lists **no prerequisite** for {title} ({code}). The database does not contain all registration/seat/administrative rules, so I cannot confirm full registration eligibility.",state,retrieved
        req=prerequisite_codes(prereq); completed=set(history_df["course_code"].astype(str).map(norm_code)); missing=[x for x in req if x not in completed]
        if missing: return f"The documented prerequisite for **{title} ({code})** is **{prereq}**. The selected student history does not show completion of **{', '.join(missing)}**. Other registration rules are not available in the database.",state,retrieved
        return f"The documented prerequisite for **{title} ({code})** is **{prereq}**, and the selected student history shows the required prerequisite course(s). Other registration rules are not available in the database.",state,retrieved

    return INSUFFICIENT,state,retrieved


# ============================================================
# VISUAL-FIRST VIDYASHILP AI ADVISOR
# ============================================================

import base64

# ---------- LOGO ----------
logo_html = ""
if LOGO:
    b64 = base64.b64encode(LOGO.read_bytes()).decode()
    logo_html = f'<img src="data:image/png;base64,{b64}" class="vu-logo" alt="Vidyashilp University">'
else:
    logo_html = '<div class="vu-logo-fallback">V</div>'

# ---------- VISUAL SYSTEM ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}
.block-container {
    max-width: 1120px;
    padding: .55rem 1.2rem 1.5rem;
}
header[data-testid="stHeader"] { background: transparent; }

/* Remove dashboard clutter */
div[data-testid="stMetric"],
[data-testid="stExpander"],
div[data-testid="stCaptionContainer"] {
    display: none !important;
}
section[data-testid="stSidebar"] {
    background: #f7f9fc;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

/* ---------- TOP BAR ---------- */
.topbar {
    height: 66px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    border-bottom:1px solid #edf0f5;
    margin-bottom:18px;
}
.brand {
    display:flex;
    align-items:center;
    gap:12px;
}
.vu-logo {
    width:46px;
    height:46px;
    object-fit:contain;
}
.vu-logo-fallback {
    width:46px;height:46px;border-radius:14px;
    display:flex;align-items:center;justify-content:center;
    background:#123f91;color:white;font-size:24px;font-weight:800;
}
.brand-name {
    font-family:Manrope,sans-serif;
    color:#123b86;
    font-weight:800;
    font-size:17px;
    letter-spacing:-.3px;
}
.online {
    display:flex;align-items:center;gap:7px;
    font-size:11px;color:#16803c;
    padding:7px 11px;border:1px solid #bce8ca;
    border-radius:999px;background:#f5fff7;
}
.online-dot {
    width:7px;height:7px;border-radius:50%;background:#22c55e;
    box-shadow:0 0 0 4px #dcfce7;
}

/* ---------- HERO / ROBOT STAGE ---------- */
.stage {
    position:relative;
    min-height:430px;
    overflow:hidden;
    border-radius:30px;
    background:
        radial-gradient(circle at 50% 43%, rgba(90,142,255,.22), transparent 22%),
        radial-gradient(circle at 50% 55%, rgba(49,85,180,.20), transparent 45%),
        linear-gradient(145deg,#071b42 0%,#0d2f72 48%,#183f91 100%);
    box-shadow:0 24px 70px rgba(18,59,134,.18);
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
}
.orbit {
    position:absolute;
    border:1px solid rgba(147,197,253,.20);
    border-radius:50%;
    pointer-events:none;
}
.orbit.one { width:360px;height:360px; }
.orbit.two { width:470px;height:470px; }
.orbit.three { width:600px;height:600px; opacity:.5; }
.float-dot {
    position:absolute;width:7px;height:7px;border-radius:50%;
    background:#93c5fd;box-shadow:0 0 16px #60a5fa;
    animation:float 5s ease-in-out infinite;
}
.d1{top:18%;left:19%}.d2{top:25%;right:17%;animation-delay:1s}
.d3{bottom:20%;left:25%;animation-delay:2s}.d4{bottom:17%;right:23%;animation-delay:3s}
@keyframes float {50%{transform:translateY(-14px);opacity:.55}}

.robot {
    position:relative;
    width:180px;height:220px;
    z-index:3;
    animation:breathe 4s ease-in-out infinite;
}
@keyframes breathe {50%{transform:translateY(-5px)}}
.robot-hair {
    position:absolute;left:22px;top:8px;width:136px;height:90px;
    background:linear-gradient(145deg,#dbeafe,#93c5fd);
    border-radius:72px 72px 42px 42px;
    box-shadow:inset -10px -8px 0 rgba(37,99,235,.12);
}
.robot-hair:after {
    content:"";position:absolute;right:-3px;top:35px;width:30px;height:52px;
    background:#bfdbfe;border-radius:0 24px 24px 22px;
}
.robot-head {
    position:absolute;left:32px;top:38px;width:116px;height:105px;
    background:linear-gradient(160deg,#ffffff,#eaf2ff);
    border:3px solid #93c5fd;border-radius:42px 42px 38px 38px;
    box-shadow:0 16px 35px rgba(0,0,0,.20);
    z-index:2;
}
.robot-ear {
    position:absolute;top:74px;width:14px;height:30px;border-radius:10px;
    background:#bfdbfe;border:2px solid #93c5fd;z-index:1;
}
.robot-ear.left{left:20px}.robot-ear.right{right:20px}
.robot-eye {
    position:absolute;top:73px;width:13px;height:13px;border-radius:50%;
    background:#2563eb;box-shadow:0 0 13px rgba(37,99,235,.65);
}
.robot-eye.left{left:59px}.robot-eye.right{left:108px}
.robot-mouth {
    position:absolute;left:77px;top:101px;width:27px;height:10px;
    border-bottom:3px solid #2563eb;border-radius:0 0 18px 18px;
}
.robot-neck {
    position:absolute;left:77px;top:137px;width:26px;height:20px;
    background:#bfdbfe;border-radius:7px;z-index:1;
}
.robot-body {
    position:absolute;left:48px;top:150px;width:84px;height:64px;
    border-radius:24px 24px 17px 17px;
    background:linear-gradient(160deg,#e0edff,#b7d3ff);
    border:2px solid #8db8f6;
    box-shadow:0 14px 28px rgba(0,0,0,.17);
}
.robot-badge {
    position:absolute;left:75px;top:169px;width:30px;height:30px;border-radius:10px;
    background:#fff;border:1px solid #93c5fd;
    display:flex;align-items:center;justify-content:center;
    color:#123f91;font-weight:800;font-size:11px;z-index:4;
}
.robot-title {
    position:absolute;bottom:32px;z-index:5;
    color:white;font-family:Manrope,sans-serif;font-size:20px;font-weight:800;
    letter-spacing:-.3px;
}
.robot-sub {
    position:absolute;bottom:12px;z-index:5;
    color:rgba(255,255,255,.70);font-size:11px;
}

/* ---------- INTERACTION DOCK ---------- */
.dock {
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:10px;
    margin:14px 0 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- TOP BAR ----------
st.markdown(f"""
<div class="topbar">
  <div class="brand">
    {logo_html}
    <div class="brand-name">VIDYASHILP UNIVERSITY</div>
  </div>
  <div class="online"><span class="online-dot"></span>Online</div>
</div>
""", unsafe_allow_html=True)

# ---------- PROFILE ----------
profile_options = ["New / General User"] + students["student_id"].astype(str).tolist()
if "selected_student" not in st.session_state:
    st.session_state.selected_student = "New / General User"

pcol1, pcol2, pcol3 = st.columns([1, 1.5, 1])
with pcol2:
    selected_student = st.selectbox(
        "Profile",
        profile_options,
        index=profile_options.index(st.session_state.selected_student),
        label_visibility="collapsed",
    )
st.session_state.selected_student = selected_student

# ---------- ROBOT STAGE ----------
st.markdown("""
<div class="stage">
  <div class="orbit one"></div>
  <div class="orbit two"></div>
  <div class="orbit three"></div>
  <span class="float-dot d1"></span>
  <span class="float-dot d2"></span>
  <span class="float-dot d3"></span>
  <span class="float-dot d4"></span>

  <div class="robot" aria-label="AIRA female academic advisor">
    <div class="robot-hair"></div>
    <div class="robot-ear left"></div>
    <div class="robot-ear right"></div>
    <div class="robot-head">
      <div class="robot-eye left"></div>
      <div class="robot-eye right"></div>
      <div class="robot-mouth"></div>
    </div>
    <div class="robot-neck"></div>
    <div class="robot-body"></div>
    <div class="robot-badge">AI</div>
  </div>

  <div class="robot-title">AIRA</div>
  <div class="robot-sub">Your academic advisor</div>
</div>
""", unsafe_allow_html=True)

# ---------- CHAT STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "advisor_state" not in st.session_state:
    st.session_state.advisor_state = {}
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Keep conversation visible, but visually compact.
for msg in st.session_state.messages[-8:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- VISUAL ACTIONS ----------
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

question = st.chat_input("Ask AIRA anything about your academics…")

if st.session_state.pending_question and not question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    start = time.perf_counter()
    answer, new_state, retrieved = answer_question(
        selected_student,
        question,
        st.session_state.advisor_state
    )
    elapsed = (time.perf_counter() - start) * 1000
    st.session_state.advisor_state = new_state

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("Evidence"):
            if retrieved is not None and not retrieved.empty:
                cols = [c for c in ["source_type", "source_row", "retrieval_score", "text"] if c in retrieved.columns]
                ev = retrieved[cols].copy()
                if "retrieval_score" in ev:
                    ev["retrieval_score"] = ev["retrieval_score"].round(3)
                st.dataframe(ev, use_container_width=True, hide_index=True)
            else:
                st.write("No supporting RAG record was retrieved.")

# ---------- MINIMAL FOOTER ----------
st.markdown('<div style="text-align:center;color:#94a3b8;font-size:10px;margin-top:12px;">Vidyashilp University • AIRA</div>', unsafe_allow_html=True)
