import os

import streamlit as st

from src.document_processor import file_hash, load_pdfs, split_documents
from src.rag_chain import ask_rag
from src.vector_store import create_vectorstore


st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)


def get_secret_api_key():
    try:
        return st.secrets.get("COHERE_API_KEY", "")
    except Exception:
        return ""


def main():
    st.title("📚 Multi-PDF RAG Assistant")
    st.caption(
        "Ask questions across multiple PDFs with grounded retrieval "
        "and source-aware answers."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "collection" not in st.session_state:
        st.session_state.collection = None

    if "file_hash" not in st.session_state:
        st.session_state.file_hash = None

    with st.sidebar:
        st.header("⚙️ RAG Settings")

        default_api_key = get_secret_api_key()

        api_key = st.text_input(
            "Cohere API Key",
            type="password",
            value=default_api_key,
            help="Used for Cohere reranking and answer generation.",
        )

        if api_key:
            os.environ["COHERE_API_KEY"] = api_key

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
        )

        top_k = st.slider(
            "Initial retrieval (Top K)",
            min_value=2,
            max_value=10,
            value=6,
        )

        threshold = st.slider(
            "Similarity distance threshold",
            min_value=0.20,
            max_value=1.00,
            value=0.55,
            step=0.05,
            help=(
                "Lower values are stricter. This is cosine distance, "
                "not a confidence percentage."
            ),
        )

        use_rerank = st.checkbox(
            "Use Cohere Rerank",
            value=True,
        )

        top_n = st.slider(
            "Reranked documents",
            min_value=1,
            max_value=5,
            value=3,
        )

    uploaded_files = st.file_uploader(
        "Upload your PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info(
            "Upload at least 3 PDFs to test single-document and "
            "multi-document questions."
        )
        return

    current_hash = file_hash(uploaded_files)
    collection_name = f"rag_{current_hash[:16]}"

    if current_hash != st.session_state.file_hash:
        with st.spinner(
            "Processing PDFs and building the vector database..."
        ):
            documents = load_pdfs(uploaded_files)

            if not documents:
                st.error(
                    "No extractable text was found in the uploaded PDFs."
                )
                return

            chunks = split_documents(
                documents,
                chunk_size=500,
                chunk_overlap=100,
            )

            st.session_state.collection = create_vectorstore(
                chunks,
                collection_name=collection_name,
            )

            st.session_state.file_hash = current_hash
            st.session_state.messages = []

        st.success(
            f"Processed {len(documents)} pages into {len(chunks)} chunks."
        )

    pdf_names = [file.name for file in uploaded_files]

    selected_sources = st.multiselect(
        "Search in selected PDFs",
        options=pdf_names,
        default=pdf_names,
    )

    if not selected_sources:
        st.warning("Select at least one PDF.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📄 Sources"):
                    for source in message["sources"]:
                        st.write(
                            f"**{source['source']}** — "
                            f"Page {source['page']}"
                        )

    question = st.chat_input("Ask a question about your PDFs...")

    if question:
        if not os.environ.get("COHERE_API_KEY"):
            st.error(
                "Please provide your Cohere API key in the sidebar "
                "or Streamlit Secrets."
            )
            return

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    answer, sources = ask_rag(
                        collection=st.session_state.collection,
                        question=question,
                        temperature=temperature,
                        similarity_threshold=threshold,
                        k=top_k,
                        top_n=top_n,
                        use_rerank=use_rerank,
                        selected_sources=selected_sources,
                    )

                    st.markdown(answer)

                    if sources:
                        with st.expander("📄 Sources"):
                            for source in sources:
                                st.write(
                                    f"**{source['source']}** — "
                                    f"Page {source['page']}"
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as exc:
                    st.error(
                        "Something went wrong while processing your "
                        f"question: {exc}"
                    )


if __name__ == "__main__":
    main()
