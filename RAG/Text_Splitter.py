from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader



load_=PyPDFLoader(r"RAG\Day 1.pdf")
doc=load_.load()
split=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)
# text=split.split_text()
text=split.split_documents(doc)
print(text[0].page_content)
r_split=RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0
)
text_=r_split.split_documents(doc)
print(len(text_))
print(text_[1].page_content)