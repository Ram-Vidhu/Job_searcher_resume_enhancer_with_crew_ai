# Multi-Agent Job Recommendation and ATS-Friendly Resume Enhancement System

This project is an advanced AI-powered application that uses a **CrewAI multi-agent system** powered by **Google Gemini 2.5 Flash** to provide **personalized job recommendations** and generate an **ATS-optimized resume draft** from a user-uploaded PDF. It's deployed on Streamlit spaces: https://jobsearcherresumeenhancerwithcrewai-ng5qxw5ot2sbu9k7vqfbsk.streamlit.app/

---

## Features

- **Intelligent Resume Analysis:**  
  Agents automatically extract and summarize key details from your resume, including skills, experience, and current role.

- **Vector Database (RAG) Job Search:**  
  Performs semantic search against a local **ChromaDB** to find the most relevant job postings.

- **ATS-Friendly Resume Enhancement:**  
  Generates a revised, keyword-rich resume draft in Markdown format optimized for Applicant Tracking Systems (ATS).

- **Streamlit Web Interface:**  
  Provides a clean and interactive user interface for uploading resumes and viewing results.

- **PDF Export:**  
  Allows downloading of the enhanced resume draft as a polished PDF.

---

## Setup and Installation

### 1. Prerequisites
Ensure you have the following installed on your system:
- **Python 3.9+**
- **pip** (Python package manager)

---

### 2. API Keys

This project requires a **Google Gemini API key** to power the AI agents.

- Obtain your key from [Google AI Studio](https://aistudio.google.com/).
- Create a `.env` file in the project root and add the following line:

```bash
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

---

### 3. Data Download

The system uses a dataset of job postings to build the recommendation vector database.

Download the dataset from Kaggle:  
**[1.3M+ LinkedIn Jobs and Skills (2024)](https://www.kaggle.com/)**

You’ll need the following files:
- `job_skills.csv`
- `job_summary.csv`
- `linkedin_job_postings.csv`

---


### 4. Install Python Dependencies

Once prerequisites are installed, run:
```bash
pip install -r requirements.txt
```

---

## How to Run the Application

The setup involves two main steps: **building the job vector database** and **running the web application**.

### Step 1: Initialize the Job Vector Database (ChromaDB)

Run the **`GenAI_Project.ipynb`** notebook completely. This will:
- Process the downloaded datasets (update file paths as needed)
- Load and clean data using **PySpark**
- Generate embeddings for job postings
- Create a persistent `chroma_db` directory containing the vector store

---

### Step 2: Launch the Streamlit Web App

From your project root, run:

```bash
streamlit run app.py
```

This will launch the Streamlit interface in your browser.

Upload your **resume (PDF)** and click **"Start Analysis and Enhancement"** to initiate the multi-agent pipeline.

---

## Directory Overview

```
project_root/
├── app.py
├── crew.py
├── agents.py
├── tasks.py
├── tools.py
├── Notebook/
│   ├── GenAI_Project.ipynb
├── chroma_db/
├── data/
│   ├── job_skills.csv
│   ├── job_summary.csv
│   └── linkedin_job_postings.csv
├── requirements.txt
├── .env
└── README.md
```

---

## Tech Stack

- **Language:** Python 3.9+  
- **UI:** Streamlit
- **Agents:** CrewAI Multi-Agent Framework
- **AI Model:** Google Gemini 2.5 Flash  
- **Database:** ChromaDB (Vector Database)
- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2 (on Hugging Face)
- **Data Processing:** PySpark  
- **PDF Generation:** reportlab
- **Hosted on:** Streamlit spaces

