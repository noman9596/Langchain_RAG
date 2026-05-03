text="""Topic: Understanding First Sexual Experience in a Healthy and Responsible Way

First sexual experiences are an important and personal part of human development. It is essential for students to understand that this experience is not just physical, but also emotional and psychological.

1. Consent Comes First
Both partners must willingly agree without pressure, fear, or manipulation. Consent should be clear, enthusiastic, and ongoing. Either person has the right to say no at any time.

2. Emotional Readiness
Students should understand that being physically ready is not the same as being emotionally ready. Feelings like trust, comfort, and mutual respect are key before entering any intimate relationship.

3. Communication
Healthy relationships depend on open and honest communication. Partners should be able to talk about boundaries, expectations, and comfort levels without embarrassment.

4. Respect and Boundaries
Each person has their own limits. Respecting personal space, values, and decisions is essential. No one should feel pressured to do anything they are unsure about.

5. Physical Awareness
Students should have a basic understanding of how their bodies work, including hygiene, protection, and safety. This includes knowledge about contraception and prevention of sexually transmitted infections (STIs).

6. Protection and Safety
Using protection (such as condoms) is important to prevent unintended pregnancies and STIs. Safe practices should always be emphasized.

7. Myths vs Reality
There are many misconceptions about “first time” experiences. It is often not like what is shown in media. It may feel awkward, and that is completely normal.

8. Emotional Aftereffects
People may feel different emotions afterward—happiness, confusion, attachment, or even regret. These feelings are normal and should be discussed openly.

9. Responsibility
Sexual activity comes with responsibility toward oneself and one’s partner—physically, emotionally, and ethically.

10. No Pressure Culture
Students should understand that there is no “right time” defined by society or peers. Choosing to wait is equally valid and should be respected."""


from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (RunnableLambda,RunnableBranch,RunnableParallel,
                                      RunnableSequence,RunnablePassthrough)
from langchain_core.output_parsers import StrOutputParser


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


note=PromptTemplate(
    template="Write simple notes:\n{text}",
    input_variables=["text"]
)

quiz=PromptTemplate(
    template="Generate 3 quiz questions:\n{text}",
    input_variables=["text"]
)

summary=PromptTemplate(
    template="Generate simple summary:\n{text}",
    input_variables=["text"]
)
parser=StrOutputParser()

def classify(x):
    text=x["text"]
    if len(text)<200:
        return {"level":"easy","text":text}
    elif len(text)<500:
        return{"level":"medium","text":text}
    else:
        return{"level":"hard","text":text}
    
    
level_select=RunnableLambda(classify)

parallel=RunnableParallel(
    {"notes":RunnableSequence(note,model,parser),
    "quiz":quiz|model|parser,
    "summary":summary|model|parser}
)
branch=RunnableBranch(
    (lambda x:x["level"]=="easy",parallel),
    (lambda x:x["level"]=="medium",parallel),
    (lambda x:x["level"]=="hard",parallel),
     parallel
)

merge=PromptTemplate(
    template="""
LEVEL: {level}

NOTES:
{notes}

QUIZ:
{quiz}

SUMMARY:
{summary}
""",
    input_variables=["level", "notes", "quiz", "summary"]
)

chain = (
    RunnablePassthrough()
    .assign(level=level_select)   
    | branch                      
    | merge
    | parser

)

result=chain.invoke({"text":text})
print(result)