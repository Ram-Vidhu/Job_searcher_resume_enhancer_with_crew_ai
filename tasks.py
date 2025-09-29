from crewai import Task
from tools import pdf_search_tool
from agents import resume_parser

parse_resume = Task(
  description=(
    """Read resume from the path {path} and extract the following details from the resume
        - Role they are looking for
        - Skills
        - summary (summary of the roles he contributed)
        - Experience (in years, sum it up across companies and categorise it as juinor, mid, senior, mid-senior based on the years of experience)
        - Last worked location
  """),
  expected_output=("""
    summarize the extracted information in JSON format:
        {
        "role": "...",
        "skills": ["...", "..."],
        "summary":
        "experience": "...",
        "last_location": "..."
        }    
    """
  ),
  tools=[pdf_search_tool],
  agent=resume_parser,
  async_execution=False,
  output_file='parsed_resume_summary.md' 
)
