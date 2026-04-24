import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load dataset
teachers = pd.read_csv("Final_Dataset.csv")

# Validate required columns
required_columns = ["Department", "Profile Text", "Link", "Name"]
if not all(col in teachers.columns for col in required_columns):
    st.error("Dataset must contain required columns.")
    st.stop()

# Remove missing values
teachers = teachers.dropna(subset=required_columns)

# Known departments
departments_list = [
    "Computer Science & Engineering", "Data Science", "Information Technology", "Cyber Security",
    "Artificial Intelligence", "Mechanical Engineering", "Electronics & Communication Engineering",
    "Biotechnology", "Master of Computer Applications (MCA)", "Master of Business Administration (MBA)",
    "Humanities & Applied Sciences", "Computer Science"
]

def find_departments(query):
    return [dept for dept in departments_list if dept.lower() in query.lower()]

def highlight_keywords(text, keywords):
    for keyword in keywords:
        text = re.sub(rf"\b({re.escape(keyword)})\b", r"**\1**", text, flags=re.IGNORECASE)
    return text

st.title("AskNIET - Intelligent Faculty Search")
query = st.text_input("Enter your question:")

if query:
    departments_in_query = find_departments(query)
    query_keywords = [word for word in query.split() if len(word) > 2]

    if "how many departments" in query.lower():
        num_departments = teachers["Department"].nunique()
        unique_departments = ", ".join(teachers["Department"].unique())
        st.write(f"There are **{num_departments}** departments: {unique_departments}.")

    elif ("how many teachers" in query.lower() or "how many faculties" in query.lower()) and "department" in query.lower():
        if departments_in_query:
            for department in departments_in_query:
                department_teachers = teachers[teachers["Department"].str.contains(department, case=False, na=False)]
                num_teachers = department_teachers.shape[0]
                st.write(f"There are **{num_teachers}** teachers in the **{department}** department:")
                for _, row in department_teachers.iterrows():
                    highlighted_expertise = highlight_keywords(row["Profile Text"], query_keywords)
                    st.markdown(f"- **{row['Name']}**  ")
                    st.markdown(f"  - Department: **{row['Department']}**")
                    st.markdown(f"  - Expertise: {highlighted_expertise}")
                    st.markdown(f"  - [Contact Link]({row['Link']})")
        else:
            st.warning("No specific department mentioned. Please clarify.")

    else:
        st.write("Proceeding with expertise-based search...")
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([query] + teachers["Profile Text"].tolist())
        scores = cosine_similarity(vectors[0], vectors[1:]).flatten()
        ranked_teachers = sorted(zip(teachers["Name"], teachers["Department"], teachers["Profile Text"], teachers["Link"], scores),
                                 key=lambda x: x[4], reverse=True)
        top_teachers = ranked_teachers[:3]
        
        for teacher in top_teachers:
            highlighted_expertise = highlight_keywords(teacher[2], query_keywords)
            st.write("---")
            st.markdown(f"**Best Match: {teacher[0]}**")
            st.markdown(f"- Department: **{teacher[1]}**")
            st.markdown(f"- Expertise: {highlighted_expertise}")
            st.markdown(f"- [Contact Link]({teacher[3]})")
            #st.write(f"- Similarity Score: {teacher[4]:.2f}")

