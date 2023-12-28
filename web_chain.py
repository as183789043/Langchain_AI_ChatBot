from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import OpenAIModerationChain
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from operator import itemgetter
from langchain.memory import ConversationBufferWindowMemory
from policy_chain import memory_chain
import os 



memory = ConversationBufferWindowMemory(k=5,return_messages=True)

def web_chain(input:str,model_name:str,temperature_number,memory=memory):
    inputs = {"input": input}
    search = DuckDuckGoSearchRun(max_results=5)
    ## template
    template = """(zh-tw)turn the following user input into a search query for a search engine:

    chat_history:{chat_history}

    input:{input}

    ouptut:"""
    prompt = ChatPromptTemplate.from_template(template)

    web_model = ChatOpenAI(model=model_name,temperature=temperature_number)
    # chain
    search_chain = {"chat_history":RunnableLambda(memory.load_memory_variables) | itemgetter("history"),"input": RunnablePassthrough() } | prompt | web_model | StrOutputParser() | search 
    ##
    template_2 = """(zh-tw)以可愛貓咪口吻總結以下內容成4項條列式說明或執行步驟，排除任何與數字，人名相關細節 句尾加上喵喵:

    輸入內容: {input}
    輸出:"""
    prompt_2 = ChatPromptTemplate.from_template(template_2)

    #chain2
    summary_chain = prompt_2 | web_model | StrOutputParser()

    sequential_chain = (
        {"input" :search_chain}
        | summary_chain
    ) 
    return sequential_chain.invoke(inputs["input"])



if __name__=="__main__":
    inputs="車禍事故SOP"
    # response=web_chain(inputs)
    # print(response)
    # # # #memory
    # # memory.save_context(inputs, {"output": response})
    # # memory.load_memory_variables({})
    # # query="我的對話紀錄中最後一個問題什麼"
    # # inputs = {"input": query}
    # # response = memory_chain.invoke(inputs["input"])
    # # response
    # memory.save_context({"inputs":inputs}, {"output": response})
    # print(memory.load_memory_variables({}))
    # ans=memory_chain("聊天紀錄的內容")
    # print(ans)

    response=web_chain(inputs)
    # response = memory_chain.invoke(inputs["input"])
    print(response)
    # save_to_memory("紅燈右轉罰則","3600元")
    # ans=memory_chain("我的上一個問題是甚麼")
    # print(ans)
    memory.save_context({"inputs":inputs}, {"output": response})
    print(memory.load_memory_variables({}))
    ans=memory_chain("我的前一個問題是什麼",memory=memory)
    print(ans)