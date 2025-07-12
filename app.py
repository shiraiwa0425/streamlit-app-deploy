#.env
from dotenv import load_dotenv
load_dotenv()
# Streamlit
import streamlit as st
#Langchain
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.agents import AgentType, initialize_agent

input_message = ""
st.title("頭と体の問診票")

st.write("このアプリは、頭と体の健康状態を評価するための問診票です。")

selected_item = st.radio("頭と体のどちらを問診しますか?", ("頭", "体"))
st.divider()

input_message = st.text_input(label="症状を入力してください", placeholder="例: 頭痛、めまい、吐き気など")

#Langchainの設定
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

def result_chain(input_message, system_template):

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input}"),
    ])
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run(input_message)
    return result

def head_doctor(input_message):
    system_template = "あなたは頭の専門医です。患者の症状に基づいて、適切なアドバイスを提供してください。"
    result = result_chain(input_message, system_template)
    return result


def body_doctor(input_message):
    system_template = "あなたは体の専門医です。患者の症状に基づいて、適切なアドバイスを提供してください。"
    result = result_chain(input_message, system_template)
    return result


head_doctor_tool = Tool(
    name="頭の専門家です",
    func=head_doctor,
    description="頭の専門医として、患者の症状に基づいて適切なアドバイスを提供します。"
)
body_doctor_tool = Tool(
    name="体の専門家です",
    func=body_doctor,
    description="体の専門医として、患者の症状に基づいて適切なアドバイスを提供します。"
)

tools = [head_doctor_tool, body_doctor_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if st.button("アドバイスを取得"):
    if input_message:
        response = agent.run(input_message)
        st.write("アドバイス:\n", response)
    else:
        st.warning("症状を入力してください。")


