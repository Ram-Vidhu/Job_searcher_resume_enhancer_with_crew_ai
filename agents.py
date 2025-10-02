from dotenv import load_dotenv
import os
from crewai import Agent
from crewai import LLM
from tools import pdf_reader_tool, job_searcher_tool

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")


llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.7,
)

reader_agent = Agent(
    role="Reader",
    goal="Extract text from documents.",
    verbose=True,
    memory=True,
    backstory="You are an expert in extracting text from PDF documents.",
    tools=[pdf_reader_tool],
    allow_delegation=True,
    llm=llm
)

summarize_agent = Agent(
    role="Summarizer",
    goal="Summarize the content of documents",
    verbose=True,
    backstory="You are skilled at summarizing long documents into concise summaries.",
    allow_delegation=True,
    llm=llm
)

job_searcher_agent = Agent(
    role="Job Searcher",
    goal="Find relevant job listings based on resume details from the available chromadb using the provided tool.",
    verbose=True,
    backstory="You are an expert in searching and finding job listings based on resume details from the available chromadb using the provided tool. DONT give job recommendations from your own knowledge, only use the tool to search the chromadb and give results based on that.",
    tools=[job_searcher_tool],
    allow_delegation=False,
    llm=llm
)

# Agent for content generation
resume_drafting_agent = Agent(
    role="Resume Drafter and Enhancer",
    goal="Analyze job recommendations and the user's resume summary to generate a new, fully updated resume draft in a structured format.",
    verbose=True,
    backstory=(
        "You are an expert ATS (Applicant Tracking System) analyst and professional copywriter. "
        "Your job is to generate a complete, revised resume draft by integrating the recommended "
        "keywords, skills, and phrasing into the user's existing summary and experience sections."
    ),
    allow_delegation=False,
    llm=llm
)

# New Agent for final output formatting
resume_formatter_agent = Agent(
    role="Resume Formatter",
    goal="Take the structured resume draft and format it into a professional, clean Markdown document.",
    verbose=True,
    backstory=(
        "You are a meticulous document layout specialist. You ensure the final resume draft "
        "is perfectly structured using professional Markdown, ready to be converted to PDF."
    ),
    allow_delegation=False,
    llm=llm
)

# # New Agent for final output consolidation
# aggregator_agent = Agent(
#     role="Output Aggregator",
#     goal="Collect the job recommendations and the enhanced resume content and combine them into a single, comprehensive JSON object.",
#     verbose=True,
#     backstory=(
#         "You are a meticulous data consolidator. Your job is to ensure all final results "
#         "from the crew—the recommended jobs and the final enhanced resume—are packaged "
#         "into a single, easy-to-parse JSON structure."
#     ),
#     allow_delegation=False, # Keep it focused on formatting the final output
#     llm=llm
# )
