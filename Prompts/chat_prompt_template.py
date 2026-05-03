# from langchain_core.prompts import ChatPromptTemplate

# template=ChatPromptTemplate([
#     ("system","you are {domain} teacher that expert in your domain"),
#     ("user","explain me these {terms} in 10 lines")
# ]
# )

# prompt=template.invoke({"domain":"sceince","terms":"education"})

# print(prompt)


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful tutor."),
    MessagesPlaceholder(variable_name="history"),  # 👈 IMPORTANT
    ("user", "{input}")
])


history = []

while True:
    user = input("You: ")

    if user.lower() == "exit":
        print("Bot: Bye!")
        break

    # add user message
    history.append(HumanMessage(content=user))

    # build prompt
    prompt = chat_prompt.invoke({
        "input": user,
        "history": history
    })

    # get response
    result = model.invoke(prompt)

    answer = result.content

    # add bot response
    history.append(AIMessage(content=answer))

    print("Bot:", answer)