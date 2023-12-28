import streamlit as st
import os 
from langchain.memory import ConversationBufferWindowMemory
from policy_chain import memory_chain
from web_chain import web_chain
from other import generate_image , display_image
from router import router_rule
from PIL import Image
import requests
from io import  BytesIO

memory = ConversationBufferWindowMemory(k=5,return_messages=True)

with st.sidebar:
    st.header("大語言模型實作結訓專題:第八組")
    text_model_option = st.selectbox(
    "挑選文字處理模型",
    ["gpt-4","gpt-4-32k","gpt-3.5-turbo","gpt-3.5-turbo-16k","gpt-3.5-turbo-1106"],index=0)
    st.write('You selected: ', text_model_option)
    text_temperature=st.slider("Temprature 影響文字模型解析的創意程度",0,10,0)

    image_model_option = st.selectbox(
    "挑選影像處理模型",
    ["dall-e-3","dall-e-2"],index=0)
    st.write('You selected: ', image_model_option)

    os.environ["OPENAI_API_KEY"] = st.text_input("請輸入OpenAI KEY",type='password')
    os.environ["OPENAI_ORGANIZATION"] = st.text_input("請輸入OpenAI organization KEY",type='password')

if os.environ["OPENAI_API_KEY"] and os.environ["OPENAI_ORGANIZATION"]:


    st.title("AI Chat bot")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])



    # Accept user input
    if prompt := st.chat_input("請輸入問題!"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner('回應內容生成中...'):
            with st.chat_message("ai"):
                ans={}
                router=router_rule(prompt)
                if router=="是":
                    response = memory_chain(prompt,text_model_option,memory)
                    ans['response']=response
                    st.markdown((ans['response']))
                   
        
                else:
                    response=web_chain(prompt,text_model_option,text_temperature,memory)
                    ans['response']=response
                    st.markdown(ans["response"])
                   
                    with st.spinner('貓咪圖片生成中...'):    
                        image_url = generate_image(ans['response'],image_model_name=image_model_option)
                        response = requests.get(image_url)
                        img = Image.open(BytesIO(response.content))
                        cat_image=st.image(img, caption=st.write(f"<a href='{image_url}'>貓咪圖片連結</a>",unsafe_allow_html=True), use_column_width=True)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "ai", "content": ans["response"]})
        
else:
    st.warning("輸入Key 開始進行聊天")


