import streamlit as st
import os

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline


st.title("FAST University Chatbot (RAG)")

#  URLs
urls = [
    "https://pwr.nu.edu.pk/about/",
    "https://pwr.nu.edu.pk/campus-facilities/",
    "https://nu.edu.pk/Admissions/HowToApply",
    "https://nu.edu.pk/Admissions/FeeStructure",
    "https://nu.edu.pk/Admissions/Scholarship",
    "https://pwr.nu.edu.pk/cs-faculty/"
]

#  Embeddings (load once)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

#  LLM (load once)
@st.cache_resource
def get_llm():
    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=200
    )
    return HuggingFacePipeline(pipeline=pipe)

#  Create or Load Vector Store
@st.cache_resource
def get_vector_store():
    embeddings = get_embeddings()

    if os.path.exists("uni_index"):
        st.info("Loading existing FAISS index")
        return FAISS.load_local(
            "uni_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

    st.warning("First time setup: scraping website")

    docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(
                url,
                requests_kwargs={"timeout": 10}
            )
            docs.extend(loader.load())
        except Exception as e:
            print(f"Failed: {url}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vector_store.save_local("uni_index")
    return vector_store


#  Build RAG Chain
@st.cache_resource
def get_rag_chain():
    vector_store = get_vector_store()
    llm = get_llm()
    parser = StrOutputParser()

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate(
    template="""
You MUST answer ONLY from the given context.
Do NOT add any external knowledge.
If the answer is not explicitly in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer (strictly from context, max 5 lines):
""",
    input_variables=["context", "question"]
)
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | parser
    )

    return rag_chain


#  Load system
rag_chain = get_rag_chain()

# Chat UI
query = st.text_input("Ask about FAST University:")

if query:
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(query)
        st.write(response)