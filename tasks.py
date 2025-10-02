from crewai import Task
from tools import pdf_reader_tool
from agents import reader_agent, summarize_agent, job_searcher_agent, resume_drafting_agent, resume_formatter_agent

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

resume_drafting_task = Task(
    description=(
        "**CRITICAL:** You must use the recommended job details from the previous task's output "
        "(Job Searcher output) to guide your enhancements. "
        "Also, retrieve the user's original resume details/summary from the **Summarizer Agent's output** "
        "(Task 2) which is available in the crew context. "
        
        "Using BOTH sets of information, generate a complete, updated resume draft. Integrate "
        "the missing high-value keywords and skills into the Summary, Experience, and Skills sections. "
        "Only output the raw text content."
    ),
    expected_output=(
        """
        A complete, structured resume draft as a raw text string, using the following key-value structure:
        "Name": "...",
        "Contact": "...",
        "Summary": "Revised professional summary with ATS keywords...",
        "Skills": ["Revised Skill 1", "New Skill 2", ...],
        "Experience": [
            {"Title": "...", "Company": "...", "Dates": "...", "Description": "Revised bullet point with ATS keywords."},
            ...
        ],
        "Education": "..."
        """
    ),
    agent=resume_drafting_agent,
)

# Format the Resume (Final step)
resume_formatting_task = Task(
    description=(
        "Take the structured resume draft from the previous task and generate a final output document "
        "using clean, professional **Markdown** formatting. Ensure proper headings, bolding, and bullet points "
        "are used to create a visually clean, single-page resume layout."
    ),
    expected_output=(
        """
        The complete, enhanced resume formatted as a single Markdown string, ready for display or PDF conversion.
        The output must ONLY be the Markdown content.
        """
    ),
    agent=resume_formatter_agent,
)

# aggregation_task = Task(
#     description=(
#         "Collect the Markdown-formatted enhanced resume from the previous task and the "
#         "Job Recommendation JSON (from the Job Searcher Task, available in context). "
#         "Combine them into a single JSON object with two top-level keys: 'jobs' and 'enhanced_resume_markdown'."
#     ),
#     expected_output=(
#         """
#         {
#             "job1": {
#                 "Job Title - Company": "...",
#                 "Location": "...",
#                 "Summary": "...",
#                 "Skills": "...",
#                 "Link": "...",
#                 "Similarity Score": "..."
#             },
#             "job2": {
#                 "Job Title - Company": "...",
#                 "Location": "...",
#                 "Summary": "...",
#                 "Skills": "...",
#                 "Link": "...",
#                 "Similarity Score": "..."
#             }....
#         }
#         """
#     ),
#     agent=aggregator_agent
# )
