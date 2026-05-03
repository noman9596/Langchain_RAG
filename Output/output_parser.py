from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.prompts import PromptTemplate


llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens":50,
        "temperature":0.3,
        "return_full_text": False
    }
)

model=ChatHuggingFace(llm=llm)
json_parser=JsonOutputParser()
parser=StrOutputParser()

template1=PromptTemplate(template="Give me 10 lines about {topic}",
                        input_variables=["topic"])

template2=PromptTemplate(template="Give me summary about {text}",
                        input_variables=["text"])

template3=PromptTemplate(
    input_variables=[],
    template="tell me about first night with wife\n{format_instruction}",
    partial_variables={"format_instruction":json_parser.get_format_instructions()}
)


chain=template1|model|parser|template2|model|parser
chain1=template3|model|json_parser
result=chain1.invoke({})
print(result)


# Tini lama not gave json format that why code fails, use higher model or StruOuputParser