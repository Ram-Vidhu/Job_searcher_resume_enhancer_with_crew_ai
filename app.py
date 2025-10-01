import streamlit as st
import os
import tempfile
import json
import pandas as pd
from crew import run_job_analysis_crew


# Function to clean and parse the LLM JSON output
def clean_and_parse_json(crew_result: str):
    """Cleans up potential markdown fences and parses the JSON result."""
    cleaned_result = crew_result.strip()

    # Check for markdown fences (e.g., ```json\n{...}\n```)
    if cleaned_result.startswith('```'):
        # Find the start of the actual JSON content after the first newline
        start = cleaned_result.find('\n')
        # Find the start of the closing fence
        end = cleaned_result.rfind('```')

        if start != -1 and end != -1 and end > start:
            # Slice between start (after newline) and end (before closing fence)
            cleaned_result = cleaned_result[start+1:end].strip()
        else:
            # Fallback if structure is unexpected, just remove surrounding fences
            cleaned_result = cleaned_result.strip('`').strip('json').strip()

    # NOTE: The final output is expected to be a dictionary of job objects.
    return json.loads(cleaned_result)


def main():
    st.set_page_config(page_title="Resume Analysis & Job Recommender 🤖", layout="wide")
    st.title("🤖 Resume Analysis & Job Recommender powered by CrewAI and Gemini")
    st.markdown("Upload your resume (PDF) to get an immediate, structured analysis and **recommended job listings** from our internal database.")

    # Check for API Key existence early for better UX
    # Switched from GROQ_API_KEY to GEMINI_API_KEY as per your agent.py
    if "GEMINI_API_KEY" not in os.environ:
        st.warning("⚠️ **Warning:** GEMINI_API_KEY environment variable is not set. The application will not be able to execute the LLM calls.")
        st.markdown("To run this, ensure your `GEMINI_API_KEY` is set in your environment.")

    uploaded_file = st.file_uploader(
        "Upload your Resume (PDF only)",
        type=["pdf"],
        help="The file will be temporarily stored for analysis and deleted immediately after processing."
    )

    output_container = st.container()

    if uploaded_file is not None:
        if st.button("Start Analysis", type="primary"):

            # 1. Create a temporary file to store the uploaded PDF
            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_path = tmp_file.name

                st.info(f"File '{uploaded_file.name}' temporarily saved for processing...")

                # 2. Run the Crew with the temporary file path
                with st.spinner("Analyzing resume content and searching for jobs (This might take a moment)..."):
                    # The crew_result is the output of the final task (job searcher)
                    crew_result = run_job_analysis_crew(temp_file_path)

                # 3. Display the result
                output_container.subheader("✅ Analysis Complete")

                if crew_result.strip() == "API Key Error":
                    output_container.error("Analysis failed. Please ensure the GEMINI_API_KEY is correctly set in your environment.")
                    return

                # Attempt to parse the JSON result for better presentation
                try:
                    # The final output is expected to be the job list JSON
                    job_recommendations_json = clean_and_parse_json(crew_result)

                    # --- RESUME SUMMARY DISPLAY (Placeholder based on common structure) ---
                    # NOTE: Since the summary is *input* to the final task,
                    # it won't be in the final output unless specifically instructed.
                    # For a great UX, you should modify your Crew/Tasks to include the summary in the final output.
                    # Since that's not done, we'll display a placeholder and the main

                    # A robust solution requires a final task to merge the summary (from summarize_text_task.output)
                    # and the jobs (from job_searcher_task.output).
                    st.markdown("## 🔎 Recommended Jobs from DB")

                    if not job_recommendations_json:
                        st.warning("No job recommendations were found or the format was empty.")
                        return

                    # Convert the JSON dictionary of jobs into a list of dictionaries for DataFrame
                    jobs_list = list(job_recommendations_json.values())

                    if jobs_list:
                        # Extract and format job details for a clean table/display
                        df = pd.DataFrame(jobs_list)

                        # Clean up Similarity Score column for display
                        if "Similarity Score" in df.columns:
                            df["Similarity Score"] = df["Similarity Score"].astype(float).map('{:.4f}'.format)

                        # Rearrange columns for better readability
                        display_cols = [
                            "Job Title - Company", "Location", "Similarity Score",
                            "Summary", "Skills", "Link"
                        ]

                        # Display the DataFrame in Streamlit
                        st.dataframe(df[display_cols],
                                     use_container_width=True,
                                     column_config={
                                         "Job Title - Company": st.column_config.TextColumn("Title & Company"),
                                         "Similarity Score": st.column_config.TextColumn("Match Score"),
                                         "Link": st.column_config.LinkColumn("Link", display_text="Apply Link")
                                     })
                    else:
                        st.warning("The job search returned an empty list of recommendations.")


                except json.JSONDecodeError as e:
                    output_container.error(f"The model returned a non-JSON formatted output or an error occurred during parsing: {e}")
                    output_container.code(crew_result, language="json")
                except Exception as e:
                    output_container.error(f"An unexpected error occurred during result processing: {e}")
                    output_container.code(crew_result, language="text")

            finally:
                # 4. Crucial: Delete the temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    st.success(f"File '{uploaded_file.name}' analysis complete and temporary file deleted.")

    else:
        # Initial prompt when no file is uploaded
        output_container.info("Awaiting PDF resume upload. Your data will be processed and deleted immediately.")


if __name__ == "__main__":
    main()