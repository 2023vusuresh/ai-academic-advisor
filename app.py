
import re
import time
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = __import__("pathlib").Path(__file__).parent
DATA = BASE

course_master = pd.read_csv(DATA/"course_master.csv")
semester_offerings = pd.read_csv(DATA/"semester_offerings.csv")
degree_requirements = pd.read_csv(DATA/"degree_requirements.csv")
students = pd.read_csv(DATA/"students.csv")
history = pd.read_csv(DATA/"student_course_history.csv")
rag_documents = pd.read_csv(DATA/"rag_documents.csv")

# ----------------------------- RAG -----------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2), lowercase=True)
rag_matrix = vectorizer.fit_transform(rag_documents["text"].astype(str))

def retrieve_rag(query, top_k=5):
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, rag_matrix)[0]
    order = scores.argsort()[::-1][:top_k]
    return rag_documents.iloc[order].assign(retrieval_score=scores[order])

# -------------------------- Utilities --------------------------
def norm_text(x):
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def lookup_course(code):
    code = str(code).strip().upper().replace(" ","")
    x = course_master[
        course_master.course_code.astype(str).str.upper().str.replace(" ","",regex=False)==code
    ]
    return None if x.empty else x.iloc[0].to_dict()

def find_courses_by_name(question):
    q = norm_text(question)
    matches = {}
    for _,r in course_master.iterrows():
        title = str(r.course_title).strip()
        if norm_text(title) and norm_text(title) in q:
            matches[str(r.course_code).strip().upper()] = {
                "course_code":str(r.course_code).strip().upper(),
                "course_title":title
            }
    return list(matches.values())

def source_summary(code):
    x = semester_offerings[
        semester_offerings.course_code.astype(str).str.upper()==str(code).upper()
    ]
    if x.empty:
        return "No semester-offering source row found."
    rows = x[["semester","source_sheet","source_row"]].drop_duplicates().head(4)
    return "; ".join(
        f"{r.source_sheet} row {r.source_row} ({r.semester})"
        for _,r in rows.iterrows()
    )

def student_history(student_id, code):
    x = history[
        (history.student_id.astype(str).str.upper()==str(student_id).upper()) &
        (history.course_code.astype(str).str.upper()==str(code).upper())
    ]
    return None if x.empty else x.iloc[0].to_dict()

# ---------------------- Final Advisor -------------------------
def advisor(student_id, question, state):
    student = students[
        students.student_id.astype(str).str.upper()==str(student_id).upper()
    ]
    if student.empty:
        return "I do not have a valid synthetic student record.", state, None
    student = student.iloc[0]

    q = str(question).strip()
    ql = q.lower()

    retrieved = retrieve_rag(q, 5)

    if state.get("waiting_for_grade"):
        m = re.search(r"\b(A\+|A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|F|FAIL)\b", q.upper())
        if not m:
            return "Please provide your grade, such as A, B+, C, or F.", state, retrieved
        grade = m.group(1)
        title = state["course_title"]
        state.clear()
        result = "failed" if grade in {"F","FAIL"} else "passed"
        return f"Based on grade {grade}, you {result} {title}.", state, retrieved

    if state.get("waiting_for_course_code"):
        m = re.fullmatch(r"[A-Za-z]{3,6}\s*\d{3,4}", q)
        if not m:
            return "Please provide the course code so I can resolve the ambiguous course.", state, retrieved
        code = m.group(0).replace(" ","").upper()
        if lookup_course(code) is None:
            return f"Course code {code} was not found in the cleaned university data.", state, retrieved
        original = state["original_question"]
        state.clear()
        return advisor(student_id, original + " " + code, state)

    if "how many credits" in ql and "graduate" in ql:
        structures = degree_requirements.academic_structure.drop_duplicates().tolist()
        return (
            "The provided data contains multiple programme structures, so I cannot determine "
            "your graduation-credit requirement from the student profile alone. Please specify "
            "your academic structure/track. Available structures: " + ", ".join(structures) + "."
        ), state, retrieved

    sm = re.search(r"(Struct_\d{4}(?:_DS)?)", q, re.I)
    if sm and "total credit" in ql:
        structure = sm.group(1)
        x = degree_requirements[
            degree_requirements.academic_structure.astype(str).str.upper()==structure.upper()
        ]
        x = x[x.component.astype(str).str.lower()=="total credits"]
        if not x.empty:
            return f"The provided degree-requirement table lists {x.iloc[0]['required_credits']} total credits for {structure.upper()}.", state, retrieved

    if "what courses" in ql and "next semester" in ql:
        next_sem = f"S{int(student.current_semester)+1}"
        n = semester_offerings[
            semester_offerings.semester.astype(str).str.upper()==next_sem
        ].course_title.nunique()
        return (
            f"The provided semester-offering data contains {n} course entries for {next_sem}. "
            "I cannot determine registration eligibility for every course from this question alone "
            "because course-specific prerequisites and programme rules may apply. Ask about a specific course."
        ), state, retrieved

    code_matches = re.findall(r"\b[A-Z]{3,6}\s*\d{3,4}\b", q.upper())
    if code_matches:
        code = code_matches[0].replace(" ","")
        course = lookup_course(code)
        matches = [] if course is None else [{"course_code":code,"course_title":course["course_title"]}]
    else:
        matches = find_courses_by_name(q)

    if not matches:
        return "I do not have sufficient information to answer this reliably from the provided university data.", state, retrieved

    if len(matches)>1:
        state["waiting_for_course_code"] = True
        state["original_question"] = q
        options = "\n".join(f"- {x['course_code']} — {x['course_title']}" for x in matches)
        return "I found multiple courses with this name. Please specify the course code:\n"+options, state, retrieved

    code = matches[0]["course_code"]
    title = matches[0]["course_title"]
    course = lookup_course(code)
    source = source_summary(code)

    if "offered" in ql and "semester" in ql:
        m = re.search(r"semester\s*(\d+)|S(\d+)", q, re.I)
        if m:
            target="S"+(m.group(1) or m.group(2))
            x=semester_offerings[
                (semester_offerings.course_code==code) &
                (semester_offerings.semester.astype(str).str.upper()==target)
            ]
            if x.empty:
                return f"{title} ({code}) is not listed in the provided semester-offering data for semester {target}.", state, retrieved
            return f"{title} ({code}) is listed for semester {target}. Source: {source}.", state, retrieved

    if any(x in ql for x in ["did i fail","did i pass","have i failed","have i passed"]):
        state["waiting_for_grade"]=True
        state["course_title"]=title
        return f"What grade did you receive in {title}? Please provide your grade so I can determine whether you passed or failed.", state, retrieved

    if "what grade" in ql or "my grade" in ql:
        h=student_history(student_id,code)
        if h is None:
            return f"I do not have a course history record for {title}.",state,retrieved
        return f"The available synthetic student record shows grade {h['grade']} for {title} ({code}). Source: synthetic student course history.",state,retrieved

    if any(x in ql for x in ["again","retake","repeat"]):
        h=student_history(student_id,code)
        if h is None:
            return f"I do not have a course history record for {title}. The retake policy is also not available in the provided data.",state,retrieved
        if str(h["status"]).lower()=="failed":
            return f"You previously failed {title}. The available university data does not specify the retake policy.",state,retrieved
        return f"The available student record shows that you did not fail {title}. The retake policy is not available in the provided university data.",state,retrieved

    if "prerequisite" in ql:
        if bool(course["has_source_conflict"]):
            vals=sorted(set(str(x) for x in semester_offerings[semester_offerings.course_code==code].prerequisite.dropna()))
            return f"The provided university sources contain conflicting prerequisite information for {title} ({code}): {'; '.join(vals)}. I cannot determine the prerequisite reliably without verification.",state,retrieved
        return f"The documented prerequisite for {title} ({code}) is {course['prerequisite']}. Source: {source}.",state,retrieved

    if any(x in ql for x in ["can i take","can i register","am i eligible"]):
        if bool(course["has_source_conflict"]):
            return f"I cannot determine eligibility for {title} because its prerequisite information is conflicting in the provided sources.",state,retrieved
        prereq=str(course["prerequisite"]).strip()
        if not prereq or prereq.upper()=="NIL":
            return f"The provided course data lists no prerequisite for {title}. However, I cannot confirm registration eligibility without the relevant offering/programme rules.",state,retrieved
        required=re.findall(r"\b[A-Z]{3,6}\d{3,4}\b",prereq.upper())
        completed=set(history[history.student_id.astype(str).str.upper()==str(student_id).upper()].course_code.astype(str).str.upper())
        missing=[x for x in required if x not in completed]
        if missing:
            return f"No. The documented prerequisite for {title} is {prereq}, and the synthetic student history does not show completion of: {', '.join(missing)}.",state,retrieved
        return f"The documented prerequisite for {title} is {prereq}, and the synthetic student history shows the required prerequisite course(s). Other registration rules are not available in the provided data.",state,retrieved

    if "credit" in ql:
        x=semester_offerings[semester_offerings.course_code==code]
        if not x.empty:
            v=sorted(set(float(a) for a in x.credits.dropna()))[0]
            return f"{title} ({code}) is listed as {v:g} credit(s). Source: {source}.",state,retrieved

    return "I do not have sufficient information to answer this reliably from the provided university data.",state,retrieved

# -------------------------- UI --------------------------
st.set_page_config(page_title="AI Academic Advisor", page_icon="🎓", layout="wide")
st.title("🎓 AI Academic Advisor")
st.caption("Grounded in the provided cleaned university data. Synthetic student profiles are used for testing.")

student_id = st.selectbox("Select synthetic student", students.student_id.astype(str).tolist())

if "state" not in st.session_state:
    st.session_state.state = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

question = st.chat_input("Ask an academic question using the course name")

if question:
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"):
        st.write(question)

    start=time.perf_counter()
    answer,new_state,retrieved=advisor(student_id,question,st.session_state.state)
    elapsed=(time.perf_counter()-start)*1000
    st.session_state.state=new_state

    st.session_state.messages.append({"role":"assistant","content":answer})
    with st.chat_message("assistant"):
        st.write(answer)

        st.caption(f"Decision + retrieval time: {elapsed:.2f} ms")

        with st.expander("Evidence / RAG sources"):
            if retrieved is not None and len(retrieved):
                st.dataframe(
                    retrieved[["source_type","source_row","retrieval_score","text"]],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.write("No retrieved evidence was available.")

        with st.expander("Student context"):
            st.dataframe(
                students[students.student_id.astype(str)==str(student_id)],
                use_container_width=True,
                hide_index=True
            )

st.divider()
st.write("**Design rule:** If the provided academic data is missing or conflicting, the advisor does not invent a rule.")
