from crewai_tools import PDFSearchTool

pdf_search_tool = PDFSearchTool(
    config=dict(
        llm=dict(
            provider="google",
            config=dict(
                model="gemini-2.5-flash",
            ),
        ),
        embedder=dict(
            provider="google",
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
            ),
        ),
        # 👇 prevent it from checking for OpenAI key
        chroma=dict(
            persist_directory="chroma_storage",  # local vector DB
            collection_name="pdf_docs"
        )
    )
)
