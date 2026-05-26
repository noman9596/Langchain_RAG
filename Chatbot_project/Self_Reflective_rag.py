

from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFacePipeline

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder
import streamlit as st
import os
import re


st.set_page_config(
    page_title="Doctor Relationship PDF RAG",
    layout="wide"
)

st.title("Doctor Relationship Self-Reflective RAG")



PDF_PATH = r"D:\Data_Science\LLM\Lanchain\Chatbot_project\relationship_advice_doctor_perspective.pdf"

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )



@st.cache_resource
def get_llm():

    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=256,
        temperature=0.3,
        do_sample=True
    )

    return HuggingFacePipeline(
        pipeline=pipe
    )




@st.cache_resource
def get_reranker():

    return CrossEncoder(
        'cross-encoder/ms-marco-MiniLM-L-6-v2'
    )


@st.cache_resource
def get_vector_store():

    embeddings = get_embeddings()

    if os.path.exists("doctor_index"):

        st.info("Loading FAISS Index...")

        return FAISS.load_local(
            "doctor_index",
            embeddings,
            allow_dangerous_deserialization=True
        )


    st.warning("Creating FAISS index from PDF...")

    loader = PyPDFLoader(PDF_PATH)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    st.write(f"Created {len(chunks)} chunks")

   

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vector_store.save_local("doctor_index")

    st.success("FAISS index created")

    return vector_store



parser = StrOutputParser()


ANSWER_PROMPT = PromptTemplate(
    template="""
You are a medical document assistant.

STRICT RULES:
1. Answer ONLY from the context.
2. Do NOT use outside knowledge.
3. If answer is not present say:
   "I don't know based on provided context."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=[
        "context",
        "question"
    ]
)



REFLECTION_PROMPT = PromptTemplate(
    template="""
You are an expert RAG evaluator.

Evaluate the answer carefully.

Metrics:

1. Faithfulness (0-10)
2. Context Relevance (0-10)
3. Hallucination Risk

Return STRICTLY in this format:

Faithfulness: <score>
Context Relevance: <score>
Hallucination Risk: <level>

Context:
{context}

Question:
{question}

Answer:
{answer}
""",
    input_variables=[
        "context",
        "question",
        "answer"
    ]
)


vector_store = get_vector_store()

llm = get_llm()

reranker = get_reranker()


def format_docs(docs):

    return "\n\n".join(
        [doc.page_content for doc in docs]
    )


def rerank_documents(question, docs):

    pairs = [
        (question, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs = sorted(
        scored_docs,
        key=lambda x: x[1],
        reverse=True
    )

    reranked_docs = [
        doc for doc, score in scored_docs
    ]

    return reranked_docs



def retrieve(question, k=8):

    docs = vector_store.similarity_search(
        question,
        k=k
    )

    docs = rerank_documents(
        question,
        docs
    )

    docs = docs[:3]

    context = format_docs(docs)

    return context



def generate_answer(question, context):

    prompt = ANSWER_PROMPT.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    final_response = parser.invoke(response)

    return final_response

def reflect(question, context, answer):

    prompt = REFLECTION_PROMPT.format(
        context=context,
        question=question,
        answer=answer
    )

    response = llm.invoke(prompt)

    reflection = parser.invoke(response)

    return reflection


def extract_scores(reflection):

    faithfulness = re.search(
        r"Faithfulness:\s*(\d+)",
        reflection
    )

    relevance = re.search(
        r"Context Relevance:\s*(\d+)",
        reflection
    )

    hallucination = re.search(
        r"Hallucination Risk:\s*(LOW|MEDIUM|HIGH)",
        reflection,
        re.IGNORECASE
    )

    faithfulness_score = (
        int(faithfulness.group(1))
        if faithfulness else 0
    )

    relevance_score = (
        int(relevance.group(1))
        if relevance else 0
    )

    hallucination_level = (
        hallucination.group(1).upper()
        if hallucination else "HIGH"
    )

    return (
        faithfulness_score,
        relevance_score,
        hallucination_level
    )


def self_reflective_rag(question):

    MAX_ITERATIONS = 3

    k = 5

    for iteration in range(MAX_ITERATIONS):

        st.write(f"# Iteration {iteration + 1}")


        context = retrieve(
            question,
            k=k
        )

        with st.expander(
            f"Retrieved Context Iteration {iteration+1}"
        ):
            st.write(context)


        answer = generate_answer(
            question,
            context
        )

        st.write("### Generated Answer")
        st.write(answer)

        reflection = reflect(
            question,
            context,
            answer
        )

        st.code(reflection)

        (
            faithfulness,
            relevance,
            hallucination
        ) = extract_scores(reflection)

        st.write(
            f"Faithfulness: {faithfulness}"
        )

        st.write(
            f"Relevance: {relevance}"
        )

        st.write(
            f"Hallucination: {hallucination}"
        )


        if (
            faithfulness >= 7
            and relevance >= 7
            and hallucination != "HIGH"
        ):

            st.success(
                "Good answer found!"
            )

            return answer

    
        st.warning(
            "Answer quality low. Improving retrieval..."
        )

        k += 5


    st.error(
        "Maximum iterations reached."
    )

    return answer