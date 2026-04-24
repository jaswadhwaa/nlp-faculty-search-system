# NLP Faculty Search System (AskNIET)

## Overview

This project is an intelligent faculty search system that allows users to query and find relevant faculty members based on their expertise using Natural Language Processing (NLP).

It uses TF-IDF vectorization and cosine similarity to match user queries with faculty profiles.

---

## Features

* Search faculty using natural language queries
* Department-based filtering
* Keyword highlighting in results
* Top 3 relevant faculty recommendations
* Interactive UI using Streamlit

---

## Tech Stack

* Python
* Pandas
* Scikit-learn (TF-IDF, Cosine Similarity)
* Streamlit
* NLP Techniques

---

## Dataset

* Faculty dataset containing:

  * Name
  * Department
  * Profile Text (expertise)
  * Contact Link

---

## How It Works

1. User enters a query
2. TF-IDF vectorization is applied
3. Cosine similarity calculates relevance
4. Top matching faculty are displayed

---

## Output

(Add screenshots here after running the app)

---

## How to Run

```bash
pip install streamlit pandas scikit-learn
streamlit run streamm.py
```

---

## Future Improvements

* Add semantic search using BERT
* Improve ranking accuracy
* Integrate real-time database

---

## Outcome

This project demonstrates the application of NLP techniques for real-world search and recommendation systems.
