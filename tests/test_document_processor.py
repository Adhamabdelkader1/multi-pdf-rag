from langchain_core.documents import Document

from src.document_processor import split_documents


def test_split_documents_preserves_metadata():
    documents = [
        Document(
            page_content="A" * 1200,
            metadata={"source": "test.pdf", "page": 3},
        )
    ]

    chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 3


def test_split_documents_returns_content():
    documents = [
        Document(
            page_content="Machine learning and NLP are useful.",
            metadata={"source": "cv.pdf", "page": 1},
        )
    ]

    chunks = split_documents(documents)

    assert len(chunks) == 1
    assert "Machine learning" in chunks[0].page_content
