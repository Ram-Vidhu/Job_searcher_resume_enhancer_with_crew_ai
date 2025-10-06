# Multi-Agent Job Recommendation and ATS-Friendly Resume Enhancement System
This project is a sophisticated AI application that utilizes a CrewAI multi-agent system powered by Google Gemini 2.5 Flash to provide personalized job recommendations and generate an ATS-optimized resume draft from a user-uploaded PDF.

## Features
Intelligent Resume Analysis: Agents read and summarize key details from your resume (skills, experience, role).

Vector Database (RAG) Job Search: Semantic search is performed against a local ChromaDB to find the most relevant job postings.

ATS-Friendly Enhancement: An agent drafts a revised resume in professional Markdown, incorporating keywords from recommended job listings.

Streamlit Web Interface: A clean, interactive UI for uploading your resume and viewing results.

PDF Download: Ability to download the enhanced resume draft as a PDF.

## Setup and Installation
1. Prerequisites
Before starting, ensure you have Python 3.9+ and pip installed.

2. API Keys
You need a Google Gemini API Key to run the LLM agents.

Get your key from Google AI Studio.

Update the .env file in the root directory with your key:

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

3. Data Download
The project relies on a job postings dataset to build the recommendation vector database.

Download the dataset from Kaggle:
[Link](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024/data)

You will need the following files from the download:

job_skills.csv

job_summary.csv

linkedin_job_postings.csv

4. wkhtmltopdf for PDF Generation (Crucial Step)
To enable the final PDF download of the enhanced resume (via the pdfkit package), the external command-line tool wkhtmltopdf must be installed on your system before installing Python dependencies.

Operating System	Installation Command(s)
Linux (Debian/Ubuntu)	1. sudo apt update 2. sudo apt-get install wkhtmltopdf
Windows	1. Download the installer from the official site: [Link](https://wkhtmltopdf.org/downloads.html) 2. Run the installer and ensure the installation path is added to your system's PATH environment variable.

5. Dependency Installation
Install all required Python packages:

pip install -r requirements.txt

How to Run the Application
The setup requires two main steps: data ingestion (to build the job database) and application execution.

Step 1: Initialize the Job Vector Database (ChromaDB)
Execute the GenAI_Project.ipynb notebook completely. This will:

Locate and process the downloaded datasets (update paths if running locally).

Load and process the data using PySpark.

Generate embeddings for job postings.

Create a persistent chroma_db directory containing the job vector store.

Step 2: Launch the Streamlit Application
From your terminal in the project's root directory, execute the app.py file:

streamlit run app.py

Your browser will open to the Streamlit application.

Upload your resume (PDF) and click "Start Analysis and Enhancement" to kick off the multi-agent crew.
