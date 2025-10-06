from crewai import Task
from tools import pdf_reader_tool
from agents import reader_agent, summarize_agent, job_searcher_agent, resume_formatter_agent, aggregator_agent

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
    description=(
        "**YOU MUST USE THE PROVIDED TOOL** with the resume details to search the chromadb. "
        "The tool returns the job listings as a single JSON string. "
        "Your final output MUST be the **RAW, UNMODIFIED JSON STRING** returned by the tool."
    ),
    expected_output=(
        """
        A single JSON string containing the job results in the following structure (must match the tool's output format exactly, do not add any wrapping text or markdown fences):
        [
            {
                "Job Title - Company": "...",
                "Location": "...",
                "Similarity Score": "..."
                // ... all fields returned by the tool
            },
            // ...
        ]
        """
    ),
    agent=job_searcher_agent
)

resume_drafting_task = Task(
    description=(
        "Take the structured resume draft from the previous task and generate a final output document "
        "using clean, professional **Markdown** formatting. Ensure proper headings, bolding, and bullet points "
        "are used to create a visually clean, single-page resume layout."
        "The output must ONLY be the Markdown content of the enhanced resume."
    ),
    expected_output=(
        """
        The complete, enhanced resume formatted as a single Markdown string, ready for display or PDF conversion.
        """
    ),
    agent=resume_formatter_agent,
)

# # Format the Resume (Final step)
# resume_formatting_task = Task(
#     description=(
#         "Take the structured resume draft from the previous task and generate a final output document "
#         "using clean, professional **Markdown** formatting. Ensure proper headings, bolding, and bullet points "
#         "are used to create a visually clean, single-page resume layout."
#     ),
#     expected_output=(
#         """
#         The complete, enhanced resume formatted as a single Markdown string, ready for display or PDF conversion.
#         The output must ONLY be the Markdown content.
#         """
#     ),
#     agent=resume_formatter_agent,
#     output_file="enhanced_resume.md"
# )

aggregation_task = Task(
    description=(
        "Collect the Markdown-formatted enhanced resume from the previous task and the "
        "Job Recommendation JSON (from the Job Searcher Task, available in context). "
        "Combine them into a single JSON object with two top-level keys: 'jobs' and 'enhanced_resume_markdown'."
    ),
    expected_output=(
        """
        A single JSON object strictly in the following format:
        {
            "jobs":{
                "job1": {
                "Job Title - Company": "...",
                "Location": "...",
                "Summary": "...",
                "Skills": "...",
                "Link": "...",
                "Similarity Score": "..."
            },
"enhanced_resume_markdown": "### [User Name]... (Markdown content here)"
        }
        """
    ),
    agent=aggregator_agent
)