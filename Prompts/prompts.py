from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 1,
        "max_new_tokens": 100
    }
)

model=ChatHuggingFace(llm=llm)

book=input("Enter book name")
lines=input("write line that you want")

# Result=model.invoke(f"Gave me {lines} of this {book}")



template=PromptTemplate(
         input_variabels=["book","lines"],
         template="""
You are a books and quotes assistant.

Task:
Give a famous or meaningful quote from the book: {book}
User request: {lines}

Rules:
- Only return real quotes if possible
- If not sure, say you are not sure
- Do not add extra explanation

Answer:
"""

)
prompt=template.invoke(
    {
    "book":book,
    "lines":lines
    }
)
result=model.invoke(prompt)


print(result.content.strip())
