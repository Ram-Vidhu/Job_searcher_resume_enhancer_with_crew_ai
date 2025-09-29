from crewai import Agent, LLM
from dotenv import load_dotenv
from tools import pdf_search_tool
import os

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["MODEL_NAME"]="gemini-2.5-flash"

llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.7,
)

resume_parser=Agent(
    role='Resume Parser',
    goal= ("""
        Read resume from the path {path} and extract the following details
        - Role they are looking for
        - Skills
        - summary (summary of the roles he contributed)
        - Experience (in years, sum it up across companies and categorise it as juinor, mid, senior, mid-senior based on the years of experience)
        - Last worked location
    """),
    verbose=True,
    memory=True,
    backstory=(
        "You are a resume parser Agent."
        "You can read the resume and extract the required information in the specified format."
    ),
    tools=[pdf_search_tool],
    allow_delegation=False,
    llm=llm
)
