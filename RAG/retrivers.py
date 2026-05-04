from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import MultiQueryRetriever


all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
    )

vector_store=FAISS.from_documents(
    documents=all_docs,
    embedding=embeddings,
    )

query="How to improve energy levels and maintain balance?"
retriver_sim=vector_store.as_retriever(search_type="similarity",kwargs={"k":5})
retriver_mmr=vector_store.as_retriever(search_type="mmr",kwargs={"k":5,"lamba_mult":0.5})
# retriver_mqr=MultiQueryRetriever.from_llm(retriever=vector_store.as_retriever(search_kwargs={"k":5}),
#                                           llm=llm)
result=retriver_mmr.invoke(query)

for i,doc in enumerate(result):
    print(f"----result{i+1}----")
    print(doc.page_content)