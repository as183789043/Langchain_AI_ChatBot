
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import CacheBackedEmbeddings


def load_doc_to_embedding():
    # loader  = PyPDFDirectoryLoader(r"C:\Users\2200908.SYSTEX\Desktop\aischool\venv\documents")
    loader  = PyPDFDirectoryLoader(r"./documents")
    docs = loader.load()


    embeddings_model = OpenAIEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 50,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.create_documents([str(docs)])

    store = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings_model, store, namespace=embeddings_model.model
    )

    db = FAISS.from_documents(texts, cached_embedder)
    retriever = db.as_retriever()
    return retriever


