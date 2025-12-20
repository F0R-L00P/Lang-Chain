# -----------------------------------------------------------
# ------------------------- Data Loader ---------------------
# -----------------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader

# Document loading
loader = PyPDFLoader(r"CPTheory.pdf")

data = loader.load()
print(data[10])
print(data[10].page_content)
print(data[10].metadata)

# -----------------------------------------------------------
# ---------------------- Text Splitter ----------------------
# -----------------------------------------------------------
# parsing the document into smaller chunks
quote = """In publishing and graphic design, Lorem ipsum is a placeholder text commonly used to demonstrate the visual form of a document or a typeface without relying on meaningful content. Lorem ipsum may be used as a placeholder before the final copy is available.
The passage is attributed to an unknown typesetter in the 15th century who is thought to have scrambled parts of Cicero's De Finibus Bonorum et Malorum for use in a type specimen book. It usually begins with:"""

from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# the seperator influences how the text is chunked
# even chunk size is defined, seperator may cause chunks to be of different sizes
# specify space, but possibly allow same size splits
text_splitter = CharacterTextSplitter(separator=" ", chunk_size=24, chunk_overlap=3)

docs = text_splitter.split_text(quote)
print(docs)

# It tries to split text using increasingly “weaker” separators
# until chunks fit the size constraint.
recursive_spliter = RecursiveCharacterTextSplitter(
    chunk_size=24,
    chunk_overlap=3,
    separators=[
        "\n\n",  # "\n\n" (paragraph breaks)
        "\n",  # "\n" (line breaks)
        ". ",  # ". " (sentence endings)
        " ",  # " " (spaces)
        "",
    ],  # "" (character-by-character)
)

docs = recursive_spliter.split_text(quote)
print(docs)

# -----------------------------------------------------------
# -------------------------- RAG ----------------------------
# -----------------------------------------------------------
# import database for vectors
from langchain_chroma import Chroma

# setup embedding model
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings

loader = PyPDFLoader(r"CPTheory.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)
# List[Document] chunks
docs = text_splitter.split_documents(pages)

print(f"Loaded pages: {len(pages)}")
print(f"Created chunks: {len(docs)}")
print("Example chunk metadata:", docs[10].metadata)
print("Example chunk text preview:", docs[10].page_content[:300])

# 3) Embedding function (use model name string)
embedding_fn = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4) Build Chroma vector DB from Documents
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_fn,
    persist_directory="chroma_cp_theory",
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Test retrieval
docs = retriever.invoke("What is conformal prediction?")
print(f"Retrieved {len(docs)} documents.")
for doc in docs:
    print(
        doc.page_content[:200]
    )  # Print first 200 characters of each retrieved document

results = vectorstore.similarity_search_with_score("What is conformal prediction?", k=3)
# Lower score = more similar (for Chroma / cosine distance)
for doc, score in results:
    print(score)
    print(doc.page_content[:200])
# -----------------------------------------------------------
# ------------------------ RAG Chaining ---------------------
# -----------------------------------------------------------
import os
from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

cp_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a technical tutor for conformal prediction.\n"
            "Use ONLY the provided textbook context to answer.\n"
            "If the context does not contain the answer, say: 'Not found in the provided textbook context.'\n"
            "Always cite sources as (p. X).",
        ),
        (
            "human",
            "Question:\n{question}\n\n" "Textbook context:\n{context}\n\n" "Answer:",
        ),
    ]
)

# chain for RAG
dotenv_path = os.path.abspath(os.path.join(os.getcwd(), ".env"))
load_dotenv(dotenv_path=dotenv_path, override=True)

API_KEY = os.getenv("API_KEY")
ENDPOINT = os.getenv("ENDPOINT")
PROJECT_ID = os.getenv("PROJECT_ID")

llm = ChatWatsonx(
    model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    url=ENDPOINT,
    project_id=PROJECT_ID,
    api_key=API_KEY,
    params={"max_new_tokens": 500, "decoding_method": "greedy"},
)


def format_docs(docs):
    parts = []
    for d in docs:
        page = d.metadata.get("page")
        page_str = f"(p. {page + 1}) " if isinstance(page, int) else ""
        parts.append(page_str + d.page_content)
    return "\n\n---\n\n".join(parts)


rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | cp_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is conformal guarantee?")
print(answer)
