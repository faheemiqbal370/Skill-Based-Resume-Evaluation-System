import streamlit as st
import json
from pdfminer.high_level import extract_text
import spacy
import re
import pandas as pd

# Load spaCy model
@st.cache_resource
def load_spacy():
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        st.error("spaCy model en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
        raise
    return nlp

nlp = load_spacy()

# Load skills JSON
@st.cache_data
def load_skills(path="skill_keywords.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

skill_keywords = load_skills()

# PDF to text
def pdf_to_text(uploaded_file):
    try:
        text = extract_text(uploaded_file)
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {e}")
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Preprocess text
def preprocess(text):
    doc = nlp(text)
    tokens = [tok.lemma_.lower() for tok in doc if tok.is_alpha and not tok.is_stop]
    return " ".join(tokens)

# Extract skills (rule-based)
def extract_skills_from_text(text, skills_dict):
    text_low = text.lower()
    found = {}
    for skill, kws in skills_dict.items():
        for kw in kws:
            if kw.lower() in text_low:
                found[skill] = 1
                break
    return found

# Confidence scoring (simple heuristic)
def confidence_score(skill, text, skills_dict):
    kws = skills_dict.get(skill, [])
    hits = sum(text.lower().count(kw.lower()) for kw in kws)
    score = min(1.0, hits / (1 + len(kws)))
    return round(score, 2)

# ---------------- UI START ---------------- #
st.title("AI Resume Analyser — Rule-based Starter")
st.write("Upload a resume PDF and paste the job description. The app will show matched and missing skills.")

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Upload resume (PDF)", type=["pdf"])

with col2:
    jd_text = st.text_area("Paste Job Description (or key skills)", height=250)

if uploaded is not None:
    with st.spinner("Extracting text from PDF..."):
        resume_raw = pdf_to_text(uploaded)

    st.subheader("Resume text (first 1000 chars)")
    st.write(resume_raw[:1000] + ("..." if len(resume_raw) > 1000 else ""))

    # preprocess
    resume_prep = preprocess(resume_raw)
    jd_prep = preprocess(jd_text)

    # extract skills
    resume_skills = extract_skills_from_text(resume_raw, skill_keywords)
    jd_skills = extract_skills_from_text(jd_text, skill_keywords)

    # fallback
    if not jd_skills:
        st.warning(
            "No skills detected in job description — please paste a JD or list of skills. "
            "The app also detects skills by keywords from the sample skill list."
        )

    matched = list(set(resume_skills.keys()).intersection(set(jd_skills.keys())))
    missing = list(set(jd_skills.keys()).difference(set(resume_skills.keys())))

    st.subheader("Results")
    st.write(f"**Skills found in resume:** {', '.join(sorted(resume_skills.keys())) if resume_skills else 'None detected'}")
    st.write(f"**Skills detected in job description:** {', '.join(sorted(jd_skills.keys())) if jd_skills else 'None detected'}")

    st.markdown("---")
    st.write("### Matched skills (resume ✅ JD) and confidence heuristic:")

    if matched:
        rows = []
        for s in sorted(matched):
            rows.append({"skill": s, "confidence": confidence_score(s, resume_raw, skill_keywords)})
        df = pd.DataFrame(rows)
        st.dataframe(df)
    else:
        st.write("No matched skills found.")

    st.write("### Missing skills (present in JD but not in resume):")
    if missing:
        for s in sorted(missing):
            st.write(f"- {s}")
    else:
        st.write("No missing skills (or none detected in JD).")


    # allow download
    if jd_skills:
        out_df = pd.DataFrame({
            "skill": sorted(list(set(list(resume_skills.keys()) + list(jd_skills.keys()))))
        })
        out_df["in_resume"] = out_df["skill"].apply(lambda x: 1 if x in resume_skills else 0)
        out_df["in_jd"] = out_df["skill"].apply(lambda x: 1 if x in jd_skills else 0)
        csv = out_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="skill_match.csv", mime='text/csv')


else:
    st.info("Upload a resume PDF to start this Analyzer. You can paste a Job Description on the right.")
