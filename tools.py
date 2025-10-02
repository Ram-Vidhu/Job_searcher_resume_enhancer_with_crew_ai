from crewai.tools import BaseTool
from PyPDF2 import PdfReader
import chromadb
import json
from sentence_transformers import SentenceTransformer
from typing import Optional
import pandas as pd


class PDFReaderTool(BaseTool):
    name: str = "PDF Reader"
    description: str = "Reads the content of a PDF file and returns the text."

    def _run(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text


pdf_reader_tool = PDFReaderTool()


class JobSearcherTool(BaseTool):
    name: str = "Job Searcher"
    description: str = "Searches for job listings based on resume details."

    def _run(self, resume_details: str, top_k: int = 5, filters: Optional[dict] = None):
        # Embedding model
        sent_transform = SentenceTransformer("all-MiniLM-L6-v2")

        # Init Chroma client (persistent storage)
        client = chromadb.PersistentClient(path="./chroma_db")

        # Create or get collection
        collection = client.get_or_create_collection(
            name="jobs",
            metadata={"hnsw:space": "cosine"}  # use cosine similarity
        )

        try:
            resume_dict = json.loads(resume_details)
        except (TypeError, json.JSONDecodeError):
            resume_dict = resume_details if isinstance(resume_details, dict) else {"text": str(resume_details)}

        # Convert dict to text for embedding
        resume_text = " ".join([f"{k}: {v}" for k, v in resume_dict.items() if v])

        embedding = sent_transform.encode([resume_text])[0]

        results = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k,
            where=filters  # will be None by default
        )

        jobs = []
        for i in range(len(results["ids"][0])):
            jobs.append({
                "similarity_score": results["distances"][0][i],
                **results["metadatas"][0][i]
            })
        result = pd.DataFrame(jobs)
        result.sort_values(by="similarity_score", ascending=False, inplace=True)
        return result


job_searcher_tool = JobSearcherTool()