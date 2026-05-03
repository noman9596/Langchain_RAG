from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field

llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens":120,
        "temperature":0.3,
        "return_full_text": False
    }
)

model=ChatHuggingFace(llm=llm)


class exp(BaseModel):
    tip:str=Field(description="gave me most important tips"),
    time:int=Field(gt=1,description="tell me in numbers how much time required"),
    position:str=Field(description="tell me most statisfying position")


parser=PydanticOutputParser(pydantic_object=exp)
template = PromptTemplate(
    input_variables=[],
    template="""
Return ONLY valid JSON. No explanation. No markdown.

{format_instruction}

Topic: relationship advice
""",
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

chain=template|model|parser
result=chain.invoke({})
print(result.content)


