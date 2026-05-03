from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

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


prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser=StrOutputParser()

parallel_chain=RunnableParallel({
    "notes":prompt1|model|parser,
    "quiz":prompt2|model|parser
}
)
chain = parallel_chain | prompt3 | model | parser

text="""A married couple’s intimate life often begins with a mix of excitement, curiosity, and sometimes a bit of nervousness—especially on the first night. It’s not about perfection or meeting any unrealistic expectations, but about creating a sense of comfort and trust with each other. On that first night, many couples simply focus on talking, laughing, and easing into the new phase of their relationship. Emotional connection matters just as much as physical closeness, and taking things slowly helps both partners feel safe and respected.
As time goes on, a healthy sex life grows through open communication, patience, and understanding each other’s needs. Partners learn what makes the other feel comfortable, valued, and loved. Small gestures—like affection, kind words, and spending quality time together—play a big role in keeping intimacy strong. Physical closeness becomes more meaningful when it’s supported by emotional bonding, mutual respect, and care.
It’s also important to remember that every couple is different. There’s no “right” timeline or way things should happen. What matters most is consent, comfort, and honesty. When both partners feel heard and appreciated, their relationship—both emotionally and physically—tends to become stronger and more fulfilling over time."""

result=chain.invoke({"text":text})
print(result)