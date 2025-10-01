import os
from crewai import Crew, Process
from agents import reader_agent, summarize_agent, job_searcher_agent
from tasks import read_pdf_task, summarize_text_task, job_searcher_task


def run_job_analysis_crew(pdf_path: str):
    """
    Sets up and executes the multi-agent Crew for job analysis.

    Args:
        pdf_path (str): The path to the uploaded PDF resume.

    Returns:
        str: The final JSON output from the summarizer agent, or "API Key Error".
    """

    # Check for API Key validity before running the costly LLM process
    if not os.environ.get("GEMINI_API_KEY"):
        return "API Key Error"  # Propagate error for app.py to handle

    # Define Crew
    job_analysis_crew = Crew(
        agents=[reader_agent, summarize_agent, job_searcher_agent],
        tasks=[read_pdf_task, summarize_text_task, job_searcher_task],
        process=Process.sequential,
        # verbose=True,  # Set verbose to 2 for detailed output in the console/logs
    )

    # Kickoff the crew, passing the PDF path as input for the first task's description.
    crew_output = job_analysis_crew.kickoff(inputs={'pdf_path': pdf_path})

    return str(crew_output)
