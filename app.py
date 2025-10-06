import streamlit as st
import os
import tempfile
import json
import pandas as pd
from crew import run_job_analysis_crew
import pdfkit
import markdown


# Function to clean and parse the LLM JSON output
def clean_and_parse_json(crew_result: str):
    """
    Cleans up potential markdown fences and extraneous characters,
    then finds and parses the first complete JSON object.
    """
    cleaned_result = crew_result.strip()

    # 1. Handle Markdown fences (e.g., ```json\n{...}\n```)
    if cleaned_result.startswith('```'):
        start = cleaned_result.find('\n')
        end = cleaned_result.rfind('```')

        if start != -1 and end != -1 and end > start:
            cleaned_result = cleaned_result[start+1:end].strip()
        else:
            cleaned_result = cleaned_result.strip('`').strip('json').strip()

    # 2. Aggressively find and isolate the first JSON object
    try:
        start_index = cleaned_result.index('{')
        end_index = cleaned_result.rindex('}') + 1

        if start_index != -1 and end_index > start_index:
            json_content = cleaned_result[start_index:end_index].strip()
            return json.loads(json_content)
        else:
            raise ValueError("Could not find a valid start '{' and end '}' for JSON.")

    except ValueError:
        return json.loads(cleaned_result) # Fallback attempt


def main():
    st.set_page_config(page_title="Resume Enhancer & Recommender 🚀", layout="wide")
    st.title("🚀 ATS Resume Enhancer & Job Recommender powered by CrewAI and Gemini")
    st.markdown("Upload your resume (PDF) to get an immediate, structured analysis, **job recommendations**, and an **ATS-friendly enhanced resume draft**.")

    # ... (API Key check remains the same) ...

    if "GEMINI_API_KEY" not in os.environ:
        st.warning("⚠️ **Warning:** GEMINI_API_KEY environment variable is not set. The application will not be able to execute the LLM calls.")
        st.markdown("To run this, ensure your GEMINI_API_KEY is set in your environment.")

    uploaded_file = st.file_uploader(
        "Upload your Resume (PDF only)",
        type=["pdf"],
        help="The file will be temporarily stored for analysis and deleted immediately after processing."
    )

    if uploaded_file is not None:
        if st.button("Start Analysis and Enhancement", type="primary"):

            temp_file_path = None
            try:
                # 1. Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_path = tmp_file.name

                st.info(f"File '{uploaded_file.name}' temporarily saved for processing...")

                # 2. Run the Crew
                with st.spinner("Analyzing resume, searching jobs, and drafting enhanced resume (This might take a moment)..."):
                    crew_result = run_job_analysis_crew(temp_file_path)

                st.subheader("✅ Analysis and Enhancement Complete")

                if crew_result.strip() == "API Key Error":
                    st.error("Analysis failed. Please ensure the GEMINI_API_KEY is correctly set in your environment.")
                    return

                # 3. Attempt to parse the aggregated JSON result
                try:
                    aggregated_output = clean_and_parse_json(crew_result)
                    job_recommendations_json = aggregated_output.get("jobs", {})
                    final_resume_markdown = aggregated_output.get("enhanced_resume_markdown", "")

                    # --- 1. Display Job Recommendations (omitted for brevity, keep your existing code) ---
                    # ... (Your code to display job recommendations goes here) ...

                    st.markdown("## 🔎 Recommended Jobs from DB")
                    # Placeholder for your job display logic
                    if job_recommendations_json:
                        jobs_list = list(job_recommendations_json.values())
                        df = pd.DataFrame(jobs_list)
                        if "Similarity Score" in df.columns:
                            df["Similarity Score"] = df["Similarity Score"].astype(float).map('{:.4f}'.format)
                        display_cols = ["Job Title - Company", "Location", "Similarity Score", "Summary", "Skills", "Link"]
                        st.dataframe(df[[col for col in display_cols if col in df.columns]], use_container_width=True, column_config={
                            "Job Title - Company": st.column_config.TextColumn("Title & Company"),
                            "Similarity Score": st.column_config.TextColumn("Match Score"),
                            "Link": st.column_config.LinkColumn("Link", display_text="Apply Link")
                        })
                    else:
                        st.warning("No job recommendations were found.")

                    st.markdown("---")


                    # --- 2. Display Enhanced Resume and Download Option ---
                    st.markdown("## ✨ Your Enhanced Resume Draft")

                    if final_resume_markdown:

                        # Display the Markdown
                        st.markdown(final_resume_markdown)

                        # --- NEW PDF GENERATION & DOWNLOAD LOGIC ---
                        try:
                            # 1. Convert Markdown to HTML
                            html_content = markdown.markdown(final_resume_markdown)

                            # 2. Convert HTML to PDF using pdfkit, saving to an in-memory buffer
                            # Note: This requires wkhtmltopdf installed and in PATH.
                            pdf_bytes = pdfkit.from_string(html_content, False)

                            # 3. Create the download button
                            st.download_button(
                                label="⬇️ Download Enhanced Resume as PDF",
                                data=pdf_bytes,
                                file_name="enhanced_resume_draft.pdf",
                                mime="application/pdf",
                                help="Click to download your new, ATS-optimized resume draft."
                            )
                            st.success("PDF created successfully!")

                        except IOError as e:
                            st.error(f"PDF Conversion Failed. Is 'wkhtmltopdf' installed and in your system's PATH? Error: {e}")
                            st.code(final_resume_markdown, language="markdown")

                        # --- End of NEW PDF LOGIC ---

                        st.warning(
                            "🚨 **Important User Warning:** This is an AI-generated draft. "
                            "**You must review and verify all content** (dates, titles, skills) before using it. "
                            "Ensure the enhancements align with your professional experience and ethical standards."
                        )
                    else:
                        st.error("The Enhancement Agent returned an empty or unreadable draft.")

                except json.JSONDecodeError as e:
                    st.error(f"The model returned a non-JSON formatted output or an error occurred during parsing: {e}")
                    st.code(crew_result, language="json")
                except Exception as e:
                    st.error(f"An unexpected error occurred during result processing: {e}")
                    st.code(crew_result, language="text")

            finally:
                # 4. Crucial: Delete the temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    st.success(f"File '{uploaded_file.name}' analysis complete and temporary file deleted.")

    else:
        st.info("Awaiting PDF resume upload. Your data will be processed and deleted immediately.")


if __name__ == "__main__":
    main()