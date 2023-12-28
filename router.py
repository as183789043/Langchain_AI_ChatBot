from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory
import os 

memory = ConversationBufferWindowMemory(k=5,return_messages=True)

def router_rule(input:str):
    inputs={"input": input}
    prompt = ChatPromptTemplate.from_template("請比對以下內容是否與道路交通法規或刑罰相關， {input}  僅回答是或否")
    model = ChatOpenAI(temperature=0)

    output_parser = StrOutputParser()

    chain = {"input": RunnablePassthrough()}|prompt | model | output_parser

    return chain.invoke(inputs["input"])

# ans={}

# inputs="台北日式料理推薦"
# router=router(inputs)
# print(f"路由是否與交通規則相關:{router}")
# if router=="是":
#     response = memory_chain(inputs,memory)
#     ans['response']=response
#     print(ans['response'])

# else:
#     response=web_chain(inputs,memory)
#     ans['response']=response
#     print(ans["response"])
#     display_image(generate_image(ans['response']))

# memory.save_context({"inputs":inputs}, {"output": response})
# ans=memory_chain("我的前一個問題是什麼",memory=memory)
# print(ans)

    