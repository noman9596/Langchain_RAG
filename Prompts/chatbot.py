from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
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


template=PromptTemplate(
        input_variables=["user"],
        template="Answer the following question clearly and briefly:\n{user}\nAnswer:"
    )

chat_history=[]
chat_history_lanchain=[
    SystemMessage(content="Answer the following question clearly and briefly:\n{user}\nAnswer:")
]

while True:
    user=input("You: ")
    if user.lower()=="exit":
        print("Bot Good Bye!")
        break

    prompt=template.invoke({"user":user})

    chat_history_lanchain.append(HumanMessage(content=user))
    chat_history.append(f"user:{prompt}")

    result=model.invoke(chat_history)

    chat_history.append(f"bot:{result.content.strip()}")
    chat_history_lanchain.append(AIMessage(content=result.content.strip()))
    
    print(result.content.strip())
    
print(chat_history_lanchain)
