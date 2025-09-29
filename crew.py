from crewai import Crew, Process
from agents import resume_parser
from tasks import parse_resume


# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
  agents=[resume_parser],
  tasks=[parse_resume],
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=True
)

## start the task execution process with enhanced feedback
result=crew.kickoff(inputs={'path':'Resume/resume.pdf'})
print(result)
