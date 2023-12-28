from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from operator import itemgetter
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
import os 

from db import load_doc_to_embedding
from other import format_docs



memory = ConversationBufferWindowMemory(k=5,return_messages=True)

def memory_chain(input,model:str,memory=memory):
    Policy_template = """(zh-tw)針對以下提供的資訊來進行問題回答:
    chat_history :{chat_history}

    context :{context}

    Question: {question}


    """

    inputs = {"input": input}
    Policy_prompt = ChatPromptTemplate.from_template(Policy_template)
    model = ChatOpenAI(temperature=0,model=model)
    retriever = load_doc_to_embedding()
    compressor = LLMChainExtractor.from_llm(model)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)


    memory_chain = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough() , "chat_history":RunnableLambda(memory.load_memory_variables) | itemgetter("history")}
        | Policy_prompt 
        | model
        | StrOutputParser()
    )
    
    return memory_chain.invoke(inputs["input"])

if __name__=="__main__":
    inputs="闖紅燈罰款"
    response=memory_chain(inputs,memory=memory)
    # response = memory_chain.invoke(inputs["input"])
    print(response)
    # save_to_memory("紅燈右轉罰則","3600元")
    # ans=memory_chain("我的上一個問題是甚麼")
    # print(ans)
    memory.save_context({"inputs":inputs}, {"output": response})
    print(memory.load_memory_variables({}))
    ans=memory_chain("我的前一個問題是什麼",memory=memory)
    print(ans)