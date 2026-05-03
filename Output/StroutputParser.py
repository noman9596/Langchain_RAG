from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 50,
        "temperature": 0.3,
        "return_full_text": False
    }
)

model = ChatHuggingFace(llm=llm)

# 1. Define schema
schema = [
    ResponseSchema(name="Tip1", description="Give most important tip"),
    ResponseSchema(name="Tip2", description="Give most important tip"),
    ResponseSchema(name="Tip3", description="Give most important tip")
]

# 2. Correct parser
parser = StructuredOutputParser.from_response_schemas(schema)

# 3. Prompt
template = PromptTemplate(
    input_variables=[],
    template="""
Tell me about grroming.

{format_instructions}
""",
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# 4. Invoke
prompt = template.format()
result = model.invoke(prompt)

print(result.content)