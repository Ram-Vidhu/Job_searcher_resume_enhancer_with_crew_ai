from crewai import Task
from tools import pdf_reader_tool
from agents import reader_agent, summarize_agent, job_searcher_agent

read_pdf_task = Task(
    description="Read the content of the PDF document located at {pdf_path}.",
    expected_output="Text extracted from the PDF document.",
    tools=[pdf_reader_tool],
    agent=reader_agent,
)

summarize_text_task = Task(
    description=("""extract the following details from the text using the specified groq model:
        - Role they are looking for
        - Skills
        - summary (summary of the roles he contributed)
        - Experience (in years, sum it up across companies and categorise it as juinor, mid, senior, mid-senior based on the years of experience)
        - Last worked location"
        """),
    expected_output=("""
    summarize the extracted information as a string in the following manner:
        '"role": "...", "skills": ["...", "..."], "summary":"....", "experience": "...", "last_location": "..."'
    """),
    agent=summarize_agent,
    model="gemini/gemini-2.0-flash"
)

job_searcher_task = Task(
    description="Using the provided resume details as parameter `resume_details` to the tool and search and provide the relevant jobs from the chromadb",
    expected_output=(
        """
        Give the output strictly in the below JSON format (ensure the [Job Title] and [Company] are correctly formatted keys):
        {
            "job1": {
                "Job Title - Company": "...",
                "Location": "...",
                "Summary": "...",
                "Skills": "...",
                "Link": "...",
                "Similarity Score": "..."
            },
            "job2": {
                "Job Title - Company": "...",
                "Location": "...",
                "Summary": "...",
                "Skills": "...",
                "Link": "...",
                "Similarity Score": "..."
            }....
        }
        """
    ),
    agent=job_searcher_agent
)