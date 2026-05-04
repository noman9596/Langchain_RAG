from langchain_community.document_loaders import TextLoader,PyPDFLoader,DirectoryLoader

loader=TextLoader(r"RAG\test.txt",encoding="utf-8")
pdf_loader=PyPDFLoader(r"RAG\Day 1.pdf")

doc=loader.load()
# print(doc[0].metadata)
pdf_doc=pdf_loader.load()
# print(len(pdf_doc))
# print(pdf_doc[1].page_content)
folder=DirectoryLoader(
    path=r"D:\Data_Science\LLM",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
folder_docs=folder.load()
print(folder_docs[6].page_content)