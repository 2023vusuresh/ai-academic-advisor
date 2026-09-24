
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE / THEME
# ============================================================
st.set_page_config(
    page_title="AI Academic Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1250px;}
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #312e81 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2.25rem;}
    .hero p {margin: 0; opacity: .9; font-size: 1rem;}
    .small-note {color: #64748b; font-size: .88rem;}
    .answer-box {
        padding: 1rem 1.15rem;
        border: 1px solid #dbeafe;
        border-radius: 14px;
        background: #f8fbff;
    }
    .warning-box {
        padding: 1rem 1.15rem;
        border: 1px solid #fde68a;
        border-radius: 14px;
        background: #fffbeb;
    }
    .safe-box {
        padding: 1rem 1.15rem;
        border: 1px solid #bbf7d0;
        border-radius: 14px;
        background: #f0fdf4;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #e2e8f0;
        padding: .7rem;
        border-radius: 12px;
        background: white;
    }
    .stButton button {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# Works whether CSVs are in repo root OR /data.
# ============================================================
BASE = Path(__file__).resolve().parent
DATA = BASE / "data" if (BASE / "data").exists() else BASE

REQUIRED_FILES = [
    "course_master.csv",
    "semester_offerings.csv",
    "degree_requirements.csv",
    "students.csv",
    "student_course_history.csv",
    "rag_documents.csv",
    "minor_courses.csv",
    "structure_courses.csv",
    "prerequisite_table.csv",
]

missing_files = [f for f in REQUIRED_FILES if not (DATA / f).exists()]
if missing_files:
    st.error(
        "The application cannot start because these university data files are missing: "
        + ", ".join(missing_files)
    )
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
course_master = D["course_master"]
semester_offerings = D["semester_offerings"]
degree_requirements = D["degree_requirements"]
students = D["students"]
history = D["history"]
rag_documents = D["rag_documents"]
minor_courses = D["minor_courses"]
structure_courses = D["structure_courses"]
prerequisite_table = D["prerequisite_table"]

@st.cache_resource
def build_retriever(texts):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        sublinear_tf=True,
        min_df=1
    )
    matrix = vectorizer.fit_transform(texts.astype(str))
    return vectorizer, matrix

vectorizer, rag_matrix = build_retriever(rag_documents["text"])

# ============================================================
# NORMALIZATION / LOOKUPS
# ============================================================
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
    rows = []
    for idx in order:
        score = float(scores[idx])
        if score < threshold:
            continue
        row = rag_documents.iloc[idx].copy()
        row["retrieval_score"] = score
        rows.append(row)
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)

def course_by_code(code):
    code = norm_code(code)
    x = course_master[
        course_master["course_code"].astype(str).map(norm_code) == code
    ]
    return None if x.empty else x.iloc[0].to_dict()

def courses_by_name(question):
    q = norm_text(question)
    matches = []
    for _, row in course_master.iterrows():
        title = str(row["course_title"]).strip()
        title_n = norm_text(title)
        if title_n and title_n in q:
            matches.append({
                "course_code": norm_code(row["course_code"]),
                "course_title": title,
            })

    # Also support exact-ish user input such as "Leadership and Teamwork Skills"
    if not matches:
        for _, row in course_master.iterrows():
            title_n = norm_text(row["course_title"])
            if title_n and (
                q == title_n
                or title_n.startswith(q)
                or q.startswith(title_n)
            ):
                matches.append({
                    "course_code": norm_code(row["course_code"]),
                    "course_title": str(row["course_title"]).strip(),
                })

    unique = {}
    for item in matches:
        unique[item["course_code"]] = item
    return list(unique.values())

def student_row(student_id):
    if not student_id or student_id == "New / General User":
        return None
    x = students[
        students["student_id"].astype(str).str.upper() == str(student_id).upper()
    ]
    return None if x.empty else x.iloc[0]

def student_history_rows(student_id):
    if not student_id or student_id == "New / General User":
        return history.iloc[0:0].copy()
    return history[
        history["student_id"].astype(str).str.upper() == str(student_id).upper()
    ].copy()

def source_for_course(code):
    code = norm_code(code)
    x = semester_offerings[
        semester_offerings["course_code"].astype(str).map(norm_code) == code
    ]
    if x.empty:
        x = prerequisite_table[
            prerequisite_table["course_code"].astype(str).map(norm_code) == code
        ]
    if x.empty:
        return "No directly matched structured source row was found."
    r = x.iloc[0]
    source_sheet = r.get("source_sheet", "unavailable")
    source_row = r.get("source_row", "unavailable")
    return f"{source_sheet}, row {source_row}"

def course_records(code):
    code = norm_code(code)
    return semester_offerings[
        semester_offerings["course_code"].astype(str).map(norm_code) == code
    ].copy()

def is_nil_prerequisite(value):
    return str(value).strip().upper() in {"", "NIL", "NONE", "NAN"}

def prerequisite_codes(value):
    return [
        norm_code(x)
        for x in re.findall(r"\b[A-Z]{3,6}\s*\d{3,4}\b", str(value).upper())
    ]

def find_course_from_question(question):
    codes = re.findall(r"\b[A-Za-z]{3,6}\s*\d{3,4}\b", question)
    if codes:
        code = norm_code(codes[0])
        course = course_by_code(code)
        if course:
            return {"type": "identified", "course": {
                "course_code": code,
                "course_title": str(course["course_title"])
            }}
        return {"type": "not_found", "course": None}

    matches = courses_by_name(question)
    if len(matches) == 1:
        return {"type": "identified", "course": matches[0]}
    if len(matches) > 1:
        return {"type": "ambiguous", "course": matches}
    return {"type": "none", "course": None}


# ============================================================
# GENERAL DATABASE QUERY HELPERS
# ============================================================
def unique_clean(values):
    out=[]
    seen=set()
    for v in values:
        v=str(v).strip()
        if not v or v.lower() in {"nan","none"} or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out

def generic_database_answer(question):
    """Answer broad natural-language questions only from exact database rows.
    Returns None when the available tables do not support a reliable answer.
    """
    q=norm_text(question)
    # 1) Semester offering queries without requiring a course name.
    m=re.search(r"(?:semester|sem|s)\s*(\d+)", q)
    if m and any(k in q for k in ["course", "courses", "offered", "offer", "available"]):
        sem="S"+m.group(1)
        x=semester_offerings[semester_offerings["semester"].astype(str).str.upper()==sem]
        if not x.empty:
            rows=x[["course_code","course_title","credits","academic_structure"]].drop_duplicates()
            rows=rows.sort_values(["course_code","course_title"])
            lines=[f"- **{r.course_code}** — {r.course_title} ({r.credits:g} credit(s)) — {r.academic_structure}" for r in rows.itertuples()]
            return f"The database lists the following courses for **{sem}**:\n\n"+"\n".join(lines)
        return f"I could not find any course-offering records for **{sem}** in the provided database."

    # 2) Degree requirement / graduation requirement queries.
    if any(k in q for k in ["degree requirement","degree requirements","graduation requirement","graduation credits","credits to graduate","total credits"]):
        structures=unique_clean(degree_requirements["academic_structure"].tolist())
        sm=re.search(r"struct[_\s-]?(\d{4})(?:[_\s-]?ds)?", q, re.I)
        if sm:
            wanted="Struct_"+sm.group(1)
            x=degree_requirements[degree_requirements["academic_structure"].astype(str).str.lower()==wanted.lower()]
            if x.empty:
                return f"I could not find **{wanted}** in the degree-requirement database."
            lines=[f"- **{r.component}: {r.required_credits:g} credits**" for r in x.itertuples()]
            return f"The documented degree requirements for **{wanted}** are:\n\n"+"\n".join(lines)
        return ("The database contains multiple academic structures, so I need the academic structure "
                f"before giving a graduation-credit answer. Available structures: {', '.join(structures)}.")

    # 3) Minor queries.
    if "minor" in q:
        minors=unique_clean(minor_courses["minor"].tolist())
        matched=[m for m in minors if norm_text(m) in q]
        if matched:
            x=minor_courses[minor_courses["minor"].astype(str).map(norm_text)==norm_text(matched[0])]
            x=x[["course_code","course_title","semester","credits","batch"]].drop_duplicates().sort_values(["semester","course_code"])
            lines=[f"- **{r.course_code}** — {r.course_title} — Semester {r.semester} — {r.credits:g} credit(s)" for r in x.itertuples()]
            return f"Courses listed under the **{matched[0]}** minor:\n\n"+"\n".join(lines)

    # 4) Structure/category queries.
    structures=unique_clean(structure_courses["academic_structure"].tolist())
    sm=None
    for st in structures:
        if norm_text(st) in q:
            sm=st; break
    if sm and any(k in q for k in ["course","courses","program","core","foundation","honors","specialization","minor"]):
        x=structure_courses[structure_courses["academic_structure"].astype(str).map(norm_text)==norm_text(sm)]
        if x.empty:
            return None
        # Optional category extraction.
        cats=unique_clean(x["structure_category"].tolist())
        cat_match=next((c for c in cats if norm_text(c) in q), None)
        if cat_match:
            x=x[x["structure_category"].astype(str).map(norm_text)==norm_text(cat_match)]
        x=x[["structure_category","course_number","course_title","credits"]].drop_duplicates()
        lines=[f"- **{r.course_number}** — {r.course_title} — {r.credits:g} credit(s)" for r in x.itertuples()]
        label=f" ({cat_match})" if cat_match else ""
        return f"Courses listed in **{sm}**{label}:\n\n"+"\n".join(lines)

    return None

# ============================================================
# GROUNDED ANSWER ENGINE
# The system never invents a rule. If structured evidence is
# missing/conflicting, it explicitly says so.
# ============================================================
INSUFFICIENT = "I do not have enough information to determine this reliably from the provided university data."

def answer_question(student_id, question, state):
    q = str(question).strip()
    ql = q.lower()
    retrieved = retrieve_rag(q)

    # Broad database questions are answered directly from structured tables.
    broad_answer = generic_database_answer(q)
    if broad_answer is not None:
        return broad_answer, state, retrieved

    # ----- Follow-up: ambiguous course -----
    if state.get("awaiting_course_code"):
        code_match = re.fullmatch(r"[A-Za-z]{3,6}\s*\d{3,4}", q)
        if not code_match:
            return (
                "I found more than one course with that name. Please provide the "
                "course code, for example UCOR103 or UCOR107.",
                state,
                retrieved
            )
        code = norm_code(code_match.group())
        if course_by_code(code) is None:
            return f"Course code {code} is not present in the provided university data.", state, retrieved
        original = state.get("original_question", "")
        state.clear()
        return answer_question(student_id, f"{original} {code}", state)

    # ----- Follow-up: grade -----
    if state.get("awaiting_grade"):
        m = re.search(r"\b(A\+|A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|F|FAIL)\b", q.upper())
        if not m:
            return "Please provide the grade, such as A, B+, C, D, or F.", state, retrieved
        grade = m.group(1)
        title = state["course_title"]
        state.clear()
        if grade in {"F", "FAIL"}:
            return f"Based on the grade you provided ({grade}), the course status is failed.", state, retrieved
        return f"Based on the grade you provided ({grade}), the course status is passed.", state, retrieved

    # ----- General university-data queries -----
    if any(k in ql for k in ["graduation credits", "credits to graduate", "how many credits to graduate", "total credits required"]):
        structures = sorted(degree_requirements["academic_structure"].dropna().astype(str).unique())
        # Only answer directly when a structure is explicitly named.
        m = re.search(r"(Struct_\d{4}(?:_DS)?)", q, re.I)
        if not m:
            return (
                "The database contains multiple academic structures, so I need your "
                "academic structure to determine the graduation-credit requirement. "
                "Available structures: " + ", ".join(structures) + ".",
                state,
                retrieved
            )
        structure = m.group(1).upper()
        x = degree_requirements[
            degree_requirements["academic_structure"].astype(str).str.upper() == structure
        ]
        if x.empty:
            return f"I could not find {structure} in the provided degree-requirement data.", state, retrieved
        total = x[
            x["component"].astype(str).str.lower().str.contains("total")
        ]
        if not total.empty:
            return f"The provided degree-requirement data lists {total.iloc[0]['required_credits']} total credits for {structure}.", state, retrieved
        return (
            f"I found the degree-requirement rows for {structure}, but the data does not provide "
            "a single explicit total-credit row.",
            state,
            retrieved
        )

    # ----- Student-required queries -----
    requires_student = any(k in ql for k in [
        "did i", "have i", "my grade", "my course", "can i take", "can i register",
        "am i eligible", "my eligibility", "what should i take", "next semester",
        "retake", "repeat", "again"
    ])

    if requires_student and (student_id == "New / General User" or not student_id):
        return (
            "This is a student-specific question. Please select an existing synthetic "
            "student profile in the sidebar so I can use the student database. "
            "I will not assume personal academic information.",
            state,
            retrieved
        )

    srow = student_row(student_id)
    history_df = student_history_rows(student_id)

    # ----- Student profile summary -----
    if any(k in ql for k in ["my profile", "my details", "show my details", "student details"]):
        if srow is None:
            return "Please select an existing synthetic student profile first.", state, retrieved
        return (
            f"Student {srow['student_id']} — {srow['programme']}, batch {srow['batch']}, "
            f"semester {srow['current_semester']}, {srow['total_credits']} total credits, "
            f"minor: {srow['minor']}.",
            state,
            retrieved
        )

    # ----- Identify course -----
    course_result = find_course_from_question(q)

    if course_result["type"] == "ambiguous":
        state["awaiting_course_code"] = True
        state["original_question"] = q
        options = "\n".join(
            f"- **{x['course_code']}** — {x['course_title']}"
            for x in course_result["course"]
        )
        return (
            "I found multiple courses matching that name. To avoid giving you the "
            "wrong answer, please specify the course code:\n\n" + options,
            state,
            retrieved
        )

    if course_result["type"] == "not_found":
        return "That course code is not present in the provided university data.", state, retrieved

    course = course_result["course"]

    if course_result["type"] == "none":
        # Safe general RAG response if there is sufficiently relevant evidence.
        if retrieved.empty:
            return INSUFFICIENT, state, retrieved
        top = retrieved.iloc[0]
        score = float(top["retrieval_score"])
        if score < 0.16:
            return INSUFFICIENT, state, retrieved
        return (
            "I found relevant university information, but the question does not map "
            "to a sufficiently specific structured rule for a reliable direct answer. "
            "Please mention the course name/code, academic structure, or student profile "
            "as applicable.",
            state,
            retrieved
        )

    code = norm_code(course["course_code"])
    title = str(course["course_title"])
    master = course_by_code(code)
    offerings = course_records(code)

    # ----- Course information -----
    if any(k in ql for k in ["what is", "tell me about", "details", "information about"]) and (
        "course" in ql or title.lower() in ql
    ):
        prereq = master.get("prerequisite", "unavailable")
        credits = sorted(set(pd.to_numeric(offerings["credits"], errors="coerce").dropna().tolist()))
        credit_text = ", ".join(f"{c:g}" for c in credits) if credits else "not available"
        return (
            f"**{title} ({code})**\n\n"
            f"- Credits listed: {credit_text}\n"
            f"- Prerequisite: {prereq}\n"
            f"- Source: {source_for_course(code)}",
            state,
            retrieved
        )

    # ----- Prerequisite -----
    if "prerequisite" in ql or "pre requisite" in ql:
        vals = []
        if master is not None:
            vals.append(str(master.get("prerequisite", "")))
        vals.extend(offerings["prerequisite"].dropna().astype(str).tolist())
        normalized_vals = {norm_text(v) for v in vals if not is_nil_prerequisite(v)}

        if len(normalized_vals) > 1:
            return (
                f"The provided sources contain materially different prerequisite information "
                f"for **{title} ({code})**. I cannot determine the prerequisite reliably "
                "without verification.",
                state,
                retrieved
            )
        prereq = master.get("prerequisite", "unavailable")
        return (
            f"The documented prerequisite for **{title} ({code})** is **{prereq}**.\n\n"
            f"Source: {source_for_course(code)}",
            state,
            retrieved
        )

    # ----- Credits -----
    if "credit" in ql:
        vals = sorted(set(pd.to_numeric(offerings["credits"], errors="coerce").dropna().tolist()))
        if not vals:
            return f"I do not have a reliable credit value for {title} ({code}).", state, retrieved
        return f"**{title} ({code})** is listed as **{vals[0]:g} credit(s)**. Source: {source_for_course(code)}.", state, retrieved

    # ----- Semester offering -----
    if any(k in ql for k in ["offered", "offer", "available", "when is", "which semester"]) and ("semester" in ql or "when" in ql):
        sems = unique_clean(offerings["semester"].tolist())
        if not offerings.empty:
            structures = ", ".join(sorted(offerings["academic_structure"].dropna().astype(str).unique()))
            sem_text = ", ".join(sems)
            return (
                f"**{title} ({code})** is listed in these semesters in the provided offering data: **{sem_text}**.\n\n"
                f"Academic structure(s): {structures}\n\n"
                f"Source: {source_for_course(code)}",
                state,
                retrieved
            )
        return f"I could not find a semester-offering record for **{title} ({code})** in the provided database.", state, retrieved

    # ----- Student grade/status -----
    if any(k in ql for k in ["did i fail", "did i pass", "have i failed", "have i passed"]):
        h = history_df[
            history_df["course_code"].astype(str).map(norm_code) == code
        ]
        if h.empty:
            return f"I do not have a course-history record for **{title} ({code})**.", state, retrieved
        grade = str(h.iloc[0]["grade"])
        status = str(h.iloc[0]["status"])
        return (
            f"Your synthetic student record lists **{title} ({code})** as **{status}** "
            f"with grade **{grade}**.",
            state,
            retrieved
        )

    if "grade" in ql or "marks" in ql:
        h = history_df[
            history_df["course_code"].astype(str).map(norm_code) == code
        ]
        if h.empty:
            return f"I do not have a course-history record for **{title} ({code})**.", state, retrieved
        return f"Your synthetic student record lists grade **{h.iloc[0]['grade']}** for **{title} ({code})**.", state, retrieved

    # ----- Retake -----
    if any(k in ql for k in ["retake", "repeat", "again"]):
        h = history_df[
            history_df["course_code"].astype(str).map(norm_code) == code
        ]
        if h.empty:
            return (
                f"I do not have a course-history record for **{title} ({code})**. "
                "The retake policy is not available in the provided data.",
                state,
                retrieved
            )
        status = str(h.iloc[0]["status"]).lower()
        if "fail" in status:
            return (
                f"Your synthetic record shows that you previously failed **{title} ({code})**. "
                "The provided university data does not specify the retake policy.",
                state,
                retrieved
            )
        return (
            f"Your synthetic record does not show a failed attempt for **{title} ({code})**. "
            "The provided university data does not specify the retake policy.",
            state,
            retrieved
        )

    # ----- Student-specific eligibility -----
    if any(k in ql for k in ["can i take", "can i register", "am i eligible", "eligible for"]):
        prereq = str(master.get("prerequisite", ""))
        if is_nil_prerequisite(prereq):
            return (
                f"The provided course data lists **no prerequisite** for {title} ({code}). "
                "However, the database does not contain all registration/seat/administrative "
                "rules, so I cannot confirm full registration eligibility.",
                state,
                retrieved
            )

        req_codes = prerequisite_codes(prereq)
        completed = set(
            history_df["course_code"].astype(str).map(norm_code)
        )
        missing = [x for x in req_codes if x not in completed]

        if missing:
            return (
                f"The documented prerequisite for **{title} ({code})** is **{prereq}**. "
                f"The selected synthetic student history does not show completion of: "
                f"**{', '.join(missing)}**.\n\n"
                "This means the prerequisite requirement is not demonstrated by the available student data. "
                "Other registration rules are not available in the database.",
                state,
                retrieved
            )

        return (
            f"The documented prerequisite for **{title} ({code})** is **{prereq}**, and the "
            "selected synthetic student history shows the required prerequisite course(s). "
            "The database does not contain all other registration rules, so this is not a guarantee of registration.",
            state,
            retrieved
        )

    # ----- Minor queries -----
    if "minor" in ql:
        qn = norm_text(q)
        rows = minor_courses[
            minor_courses["course_title"].astype(str).map(norm_text).apply(lambda x: x in qn)
        ]
        if not rows.empty:
            r = rows.iloc[0]
            return (
                f"**{r['course_title']} ({r['course_code']})** is listed under **{r['minor']}** "
                f"for semester **{r['semester']}**, with **{r['credits']:g} credit(s)**.",
                state,
                retrieved
            )

    # ----- Generic grounded RAG fallback -----
    # Never invent an answer. If the retrieved evidence is strong enough, expose
    # the database records verbatim so the user can inspect the source.
    if retrieved is not None and not retrieved.empty and float(retrieved.iloc[0]["retrieval_score"]) >= 0.22:
        evidence_lines=[]
        for _, row in retrieved.head(5).iterrows():
            txt=str(row.get("text", "")).replace("\n", " | ")
            evidence_lines.append(f"- {txt}")
        return (
            "I found relevant information in the university database, but I cannot safely "
            "convert it into a more specific conclusion without an explicit rule. Here is the "
            "grounded evidence available for your question:\n\n" + "\n".join(evidence_lines),
            state,
            retrieved
        )

    # ----- Generic safe fallback -----
    return INSUFFICIENT, state, retrieved

# ============================================================
# SIDEBAR / DASHBOARD
# ============================================================
with st.sidebar:
    st.markdown("## 🎓 Academic Advisor")
    st.caption("Grounded university-data assistant")

    profile_options = ["New / General User"] + students["student_id"].astype(str).tolist()
    selected_student = st.selectbox(
        "Student profile",
        profile_options,
        help="Choose a synthetic student only for questions requiring personal academic data."
    )

    if selected_student != "New / General User":
        s = student_row(selected_student)
        st.markdown("---")
        st.markdown("### Current profile")
        st.write(f"**Programme:** {s['programme']}")
        st.write(f"**Batch:** {s['batch']}")
        st.write(f"**Semester:** {s['current_semester']}")
        st.write(f"**Credits:** {s['total_credits']}")
        st.write(f"**Minor:** {s['minor']}")

    st.markdown("---")
    st.markdown("### What I can answer")
    st.markdown(
        "- Course prerequisites\n"
        "- Course credits\n"
        "- Semester offerings\n"
        "- Student grades/status\n"
        "- Prerequisite-based eligibility\n"
        "- Retake/history questions\n"
        "- Degree-credit requirements\n"
        "- Minor/course information\n"
    )
    st.markdown("---")
    st.caption("If the database does not support an answer, I will say so rather than invent a rule.")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.advisor_state = {}
        st.rerun()

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🎓 AI Academic Advisor</h1>
    <p>Ask about courses, prerequisites, credits, semester offerings and student-specific academic information.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# DASHBOARD METRICS
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Courses", len(course_master))
c2.metric("Offering Records", len(semester_offerings))
c3.metric("Student Profiles", len(students))
c4.metric("Knowledge Records", len(rag_documents))

st.markdown("### 💬 Ask your question")

if selected_student == "New / General User":
    st.info(
        "You are using General Mode. Ask university/course questions freely. "
        "For questions about **your** grade, eligibility, history or next semester, select a synthetic student profile in the sidebar."
    )

# Quick prompts
qcols = st.columns(4)
quick_questions = [
    "What is the prerequisite for Leadership and Teamwork Skills?",
    "How many credits is Communication Skills?",
    "Is Leadership and Teamwork Skills offered in semester 2?",
    "What are the graduation credits for Struct_2024?",
]
for col, text in zip(qcols, quick_questions):
    if col.button(text, use_container_width=True):
        st.session_state.pending_question = text

if "messages" not in st.session_state:
    st.session_state.messages = []
if "advisor_state" not in st.session_state:
    st.session_state.advisor_state = {}
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input(
    "Ask anything about the university data… e.g. “What is the prerequisite for Communication Skills?”"
)

if st.session_state.pending_question and not question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    start = time.perf_counter()
    answer, new_state, retrieved = answer_question(
        selected_student,
        question,
        st.session_state.advisor_state
    )
    elapsed = (time.perf_counter() - start) * 1000
    st.session_state.advisor_state = new_state

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown(answer)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption(f"Grounded decision time: {elapsed:.1f} ms")

        if retrieved is not None and not retrieved.empty:
            with st.expander("🔎 Evidence used"):
                evidence = retrieved[["source_type", "source_row", "retrieval_score", "text"]].copy()
                evidence["retrieval_score"] = evidence["retrieval_score"].round(3)
                st.dataframe(evidence, use_container_width=True, hide_index=True)

        if selected_student != "New / General User":
            with st.expander("👤 Student context used"):
                st.dataframe(
                    students[
                        students["student_id"].astype(str) == selected_student
                    ],
                    use_container_width=True,
                    hide_index=True
                )

st.markdown("---")
st.markdown(
    '<div class="small-note">🔒 Grounding rule: the advisor uses only the provided university database and synthetic student data. '
    'When the evidence is missing, ambiguous or conflicting, it does not invent an answer.</div>',
    unsafe_allow_html=True
)
