import hashlib
import os
import tempfile

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def save_uploaded_pdf(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded_file.getbuffer())
    temp.close()
    return temp.name


def file_hash(uploaded_files):
    digest = hashlib.sha256()
    for uploaded_file in sorted(uploaded_files, key=lambda f: f.name):
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def load_pdfs(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        pdf_path = save_uploaded_pdf(uploaded_file)
        pdf = fitz.open(pdf_path)

        try:
            for page_number, page in enumerate(pdf):
                text = page.get_text("text").strip()

                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_number + 1,
                        },
                    )
                )
        finally:
            pdf.close()
            os.remove(pdf_path)

    return documents


def split_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
