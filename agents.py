from dotenv import load_dotenv
import os
from crewai import Agent
from crewai import LLM
from tools import pdf_reader_tool, job_searcher_tool

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")


llm = LLM(
    model="gemini/gemini-2.0-flash",
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
