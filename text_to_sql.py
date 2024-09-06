from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

load_dotenv("./.env", override=True)

from typing import List, Union


from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import re
from langchain.chains import create_sql_query_chain


def extract_fenced_text(text, fence="```"):
    # print(f"[extract_fenced_text] input: text: {text}")

    pattern = f"{re.escape(fence)}(?:\w+)?\s*(.*?){re.escape(fence)}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text_return = match.group(1).strip()
        # print(f"[extract_fenced_text] output: {text_return}")
        return text_return
    
    # print(f"[extract_fenced_text] output: None")
    return None


def text_to_sql_results(question):

    db = SQLDatabase.from_uri("sqlite:///text_to_sql_example.db")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    template = '''
        Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Use the following format:

        Question: "Question here"
        SQLQuery: "SQL Query to run"
        SQLResult: "Result of the SQLQuery"
        Answer: "Final answer here"

        {top_k}

        Only use the following tables:

        {table_info}.

        Question: {input}

        Output a fenced SQL query, example:
        ```sql
        SELECT * FROM financials LIMIT 10;
        ```
    '''
    prompt = PromptTemplate.from_template(template)

    chain = create_sql_query_chain(llm, db, prompt=prompt)
    response = chain.invoke({"input": question, "question": question, "dialect": db.dialect, "table_info": db.get_context(), "top_k": 10})

    sql_query = extract_fenced_text(response)

    res = db.run(sql_query)

    return res


def text_to_sql_chat(chat_history: List[Union[BaseMessage, SystemMessage]]):

    if not isinstance(chat_history[-1], HumanMessage):
        print("Last message must be a human message")
        return
    
    # convert the latest message to a question to db (needed for the cases when the user adds information toi the previous messages)
    system_prompt_full_question = SystemMessage(content="""
        You are a helpful assistant that converts the previous user message to a full question.
        
        Example dialogue:
        User: What is the weather in San Francisco?
        Assistant: The weather in San Francisco is 64 degrees Fahrenheit.
        User: now

        Example output: 
        User: What is the weather in San Francisco now?
    """)
    llm = ChatOpenAI(model="gpt-4o-mini")
    ai_response_full_question = llm.invoke([system_prompt_full_question] + chat_history)

    ai_response_sql_results = text_to_sql_results(ai_response_full_question.content)

    return ai_response_sql_results

chat_history = []

while 1:
    new_question = input("Enter a message: ")
    if new_question == "exit":
        break
    chat_history.append(HumanMessage(content=new_question))
    # print(chat_history)
    response = text_to_sql_chat(chat_history)
    print(response)

# print(type(text_to_sql_chat([HumanMessage(content="What is the weather in San Francisco?")])))
