import os

from langchain_cohere import ChatCohere, CohereRerank
from langchain_core.documents import Document

from .prompts import RAG_PROMPT
from .vector_store import search_documents


DEFAULT_LLM_MODEL = os.getenv(
    "COHERE_LLM_MODEL",
    "command-a-plus-05-2026",
)

DEFAULT_RERANK_MODEL = os.getenv(
    "COHERE_RERANK_MODEL",
    "rerank-v4.0-fast",
)


def create_llm(temperature=0.0):
    return ChatCohere(
        model=DEFAULT_LLM_MODEL,
        temperature=temperature,
        cohere_api_key=os.environ["COHERE_API_KEY"],
    )


def rerank_documents(documents, question, top_n=3):
    if not documents:
        return []

    reranker = CohereRerank(
        model=DEFAULT_RERANK_MODEL,
        top_n=min(top_n, len(documents)),
        cohere_api_key=os.environ["COHERE_API_KEY"],
    )

    langchain_docs = [
        Document(
            page_content=item["text"],
            metadata={
                "source": item["source"],
                "page": item["page"],
                "distance": item["distance"],
            },
        )
        for item in documents
    ]

    reranked = reranker.compress_documents(langchain_docs, question)

    return [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 1),
            "distance": doc.metadata.get("distance", 0.0),
        }
        for doc in reranked
    ]


def format_context(documents):
    return "\n\n".join(
        f"[Document {i} | {doc['source']} | Page {doc['page']}]\n"
        f"{doc['text']}"
        for i, doc in enumerate(documents, start=1)
    )


def ask_rag(
    collection,
    question,
    temperature=0.0,
    similarity_threshold=0.55,
    k=6,
    top_n=3,
    use_rerank=True,
    selected_sources=None,
):
    retrieved = search_documents(
        collection=collection,
        question=question,
        k=k,
        similarity_threshold=similarity_threshold,
        selected_sources=selected_sources,
    )

    if not retrieved:
        return (
            "I couldn't find this information in the uploaded documents.",
            [],
        )

    if use_rerank:
        retrieved = rerank_documents(
            documents=retrieved,
            question=question,
            top_n=top_n,
        )

    if not retrieved:
        return (
            "I couldn't find this information in the uploaded documents.",
            [],
        )

    context = format_context(retrieved)
    llm = create_llm(temperature=temperature)

    response = llm.invoke(
        RAG_PROMPT.format_messages(
            context=context,
            question=question,
        )
    )

    return response.content, retrieved
