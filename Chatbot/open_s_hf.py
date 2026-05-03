from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ["HF_HOME"] = r"D:\Data_Science\LLM\Lanchain\huggingface_cache"

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 1,
        "max_new_tokens": 100
    }
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of Pakistan")
print(result)