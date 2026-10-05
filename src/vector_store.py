import chromadb
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "chroma_db"

_model = None


def get_embedding_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def create_vectorstore(chunks, collection_name):
    """Create a persistent Chroma collection using local embeddings."""
    model = get_embedding_model()

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Avoid adding the same chunk twice if the collection already exists.
    if collection.count() == 0:
        texts = [chunk.page_content for chunk in chunks]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        ids = [f"chunk-{i}" for i in range(len(chunks))]

        metadatas = [
            {
                "source": chunk.metadata.get("source", "Unknown"),
                "page": int(chunk.metadata.get("page", 1)),
            }
            for chunk in chunks
        ]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    return collection


def search_documents(
    collection,
    question,
    k=6,
    similarity_threshold=0.55,
    selected_sources=None,
):
    model = get_embedding_model()

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    where = None

    if selected_sources:
        where = (
            {"source": selected_sources[0]}
            if len(selected_sources) == 1
            else {"source": {"$in": selected_sources}}
        )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = []

    if not results["documents"]:
        return documents

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if distance <= similarity_threshold:
            documents.append(
                {
                    "text": text,
                    "source": metadata.get("source", "Unknown"),
                    "page": metadata.get("page", 1),
                    "distance": float(distance),
                }
            )

    return documents
