from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import re
from langchain.chains import create_sql_query_chain
from general_functions import extract_fenced_text
import json


# -------------------------------------------------------------------------------------------------
# Text-to-SQL

# # create an sqllite database

# import sqlite3
# import csv
# conn = sqlite3.connect('text_to_sql_example.db')
# cursor = conn.cursor()
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS financials (
#     Segment TEXT,
#     Country TEXT,
#     Product TEXT,
#     DiscountBand TEXT,
#     UnitsSold REAL,
#     ManufacturingPrice REAL,
#     SalePrice REAL,
#     GrossSales REAL,
#     Discounts REAL,
#     Sales REAL,
#     COGS REAL,
#     Profit REAL,
#     Date TEXT,
#     MonthNumber INTEGER,
#     MonthName TEXT,
#     Year INTEGER
# )
# ''')
# # Function to clean numeric values
# def clean_numeric(value):
#     value = value.replace('$', '').replace(',', '').replace(' ', '')
#     if value == '-':
#         return 0.0
#     if value.startswith('(') and value.endswith(')'):
#         value = '-' + value[1:-1]
#     return float(value)
# # Read the CSV file and insert data into the table
# with open('Financials.csv', 'r') as file:
#     reader = csv.reader(file)
#     next(reader)  # Skip the header row
#     for row in reader:
#         row[4] = clean_numeric(row[4])  # UnitsSold
#         row[5] = clean_numeric(row[5])  # ManufacturingPrice
#         row[6] = clean_numeric(row[6])  # SalePrice
#         row[7] = clean_numeric(row[7])  # GrossSales
#         row[8] = clean_numeric(row[8])  # Discounts
#         row[9] = clean_numeric(row[9])  # Sales
#         row[10] = clean_numeric(row[10])  # COGS
#         row[11] = clean_numeric(row[11])  # Profit
#         cursor.execute('''
#         INSERT INTO financials (
#             Segment, Country, Product, DiscountBand, UnitsSold, ManufacturingPrice, SalePrice, GrossSales, Discounts, Sales, COGS, Profit, Date, MonthNumber, MonthName, Year
#         ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#         ''', row)

# # Commit the changes and close the connection
# conn.commit()
# conn.close()
# print("Database created and data inserted successfully.")



# def text_to_sql_results(question):

#     db = SQLDatabase.from_uri("sqlite:///text_to_sql_example.db")

#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#     template = '''
#         Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
#         Use the following format:

#         Question: "Question here"
#         SQLQuery: "SQL Query to run"
#         SQLResult: "Result of the SQLQuery"
#         Answer: "Final answer here"

#         {top_k}

#         Only use the following tables:

#         {table_info}.

#         Question: {input}

#         Output a fenced SQL query, example:
#         ```sql
#         SELECT * FROM financials LIMIT 10;
#         ```
#     '''
#     prompt = PromptTemplate.from_template(template)

#     chain = create_sql_query_chain(llm, db, prompt=prompt)
#     response = chain.invoke({"input": question, "question": question, "dialect": db.dialect, "table_info": db.get_context(), "top_k": 10})

#     sql_query = extract_fenced_text(response)

#     res = db.run(sql_query)

#     return res

queries = [
    "What was the total profit for each product in the year 2014?",
    "how about 2013?",
    "by country",
    # "What are the total gross sales for each country in 2014?",
    # "Which product had the highest manufacturing price in 2014?",
    # "Which segment had the highest profit margin in 2014?",
    # "What are the total sales for each month in 2014?",
]

#     # "What was the total cost of goods sold (COGS) for 'Velo' products in the United States in 2014?",
#     # "Which country had the lowest gross sales for 'Amarilla' products in 2014?"
#     # "Which month had the highest sales for the 'Carretera' product?",
#     # "How many units of 'Montana' were sold in Germany in June 2014?",
#     # "What is the average discount given for 'Paseo' products across all countries?",

# for query in queries:
#     print(f"Query: {query}")
#     res = text_to_sql_results(query)
#     print(f"Result: {res}")
#     print("")
#     print("")


# -------------------------------------------------------------------------------------------------
# Multimodal RAG

# from miltimodal_rag_preparation_functions import (
#     collect_video_youtube,
#     extract_full_text_from_diarised_transcript,
#     extract_only_word_ts_from_transcript,
#     find_phrase_timestamps,
#     process_segments,
#     segments_to_langchain_documents,
# )
# from general_functions import faiss_index


# # # Create FAISS index
# # documents = segments_to_langchain_documents('tmp/video_6f1zm-lNMGc_segments.json')
# # faiss_index(documents, index_name="faiss_index_multimodal_rag")


# # Query FAISS index
# questions = {
#   "can be answered without images": [
#     "What is the origin of Interactive Technical Manuals (ITMs), and how have they evolved since their creation?",
#     "How can ITMs benefit technicians and engineers in troubleshooting and maintaining complex equipment?"
#   ],
#   "need images to answer": [
#     "How does the ITM's schematic interface enhance the understanding of complex hydraulic systems?",
#     "What kind of information is typically included in a component sheet within the ITM system?",
#     # "What specific navigation options does the ITM interface provide to users for accessing different types of technical information?",
#     # "In the ITM system, what types of interactive elements are typically included in a component sheet to assist technicians in understanding and troubleshooting a specific part?",
#     # "How does the ITM present complex hydraulic system schematics to make them more understandable and interactive for users?"
#   ]
# }

# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# vdb = faiss_index(index_name="faiss_index_multimodal_rag")

# question = "How does the ITM's schematic interface enhance the understanding of complex hydraulic systems?"

# results_media = vdb.similarity_search(
#     question,
#     filter={"type": "media"},
# )

# results_transcript = vdb.similarity_search(
#     question,
#     filter={"type": "transcript_chunk"},
# )

# results_media_string = ""
# for res in results_media:
#     results_media_string += f"Context piece: {res.page_content} [{res.metadata}]\n--\n\n"

# results_transcript_string = ""
# for res in results_transcript:
#     results_transcript_string += f"Context piece: {res.page_content} [{res.metadata}]\n--\n\n"

# prompt_template = """
# Here's the user's question: {question}.

# Here's the context that may be relevant:
# -----
# From video transcript:
# {results_transcript_string}
# -----
# From video media:
# {results_media_string}
# -----

# Ignore all the pieces of context that don't have information that adds up to the answer to the question.

# Answer the question based on the context.

# Avoid guessing, only use the context provided.

# If you used any video media context pieces to answer the question, please add "MEDIA: [value of 'media' field from the video media context piece that you used] (without quotes and square brackets)" at the end of your answer.
# """

# from langchain_core.prompts import PromptTemplate
# prompt = PromptTemplate.from_template(prompt_template)

# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# chain = prompt | llm

# answer = chain.invoke({
#     "question": question,
#     "results_transcript_string": results_transcript_string,
#     "results_media_string": results_media_string,
# })

# print()
# print(answer.content)


# -------------------------------------------------------------------------------------------------
# Recursive RAG

# from general_functions import faiss_index
# index_name = "faiss_index_recursive_rag"


# # # create index

# # from recursive_rag_preparation_functions import create_index_for_recursive_rag
# # create_index_for_recursive_rag(directory_path="data/for_recursive_rag", index_name=index_name)


# # Query index

# questions = [
#     "Which companies provide design services for us?",
#     "What is the cost of services of CreativeMind?",
#     "Give me contacts of CreativeMind",
# ]
# question = questions[2]

# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# vdb = faiss_index(index_name=index_name)


# results = vdb.similarity_search(
#     question,
#     # filter={"type": "media"},
# )


# results_string = ""
# for i, res in enumerate(results):
#     results_string += f"Context piece {i}:\n"
#     results_string += f"{res.page_content}"
#     results_string += "\n--\n\n"

#     print(f"Result {i}:")
#     print(f"  Page content: {res.page_content}")
#     print(f"  Metadata: {res.metadata}")
#     print("")
#     print("")




# from langchain_core.messages import SystemMessage
# simplify_question_prompt = SystemMessage(
#     content="""
#         Your role is to simplify user's questions and remove any company names or agreement numbers.

#         Example:
#         Input:
#             "What are the NDA terms with RogaIKopyta?"
#         Output (a valid json):
#             {"question": "What are the NDA terms?", "company_name": "RogaIKopyta"}
#     """
# )
# from langchain.schema import HumanMessage
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# simplified_question_response = llm.invoke([
#     simplify_question_prompt,
#     HumanMessage(content=question)
# ])

# try:
#     simplified = json.loads(extract_fenced_text(simplified_question_response.content))
# except:
#     simplified = json.loads(simplified_question_response.content)

# print(f"\n\n\n simplified question: {simplified} \n\n\n")


# prompt_template = """
#     Your role is to answer user's questions related to legal agreements with their company (Innovate Spaces Corporation).

#     Here's the user's question: {question}.

#     Here are some context pieces from agreements with various companies:
#     -----
#     {context}
#     -----

#     In case you have all relevant info, answer the question in a concise manner.

#     But if relevant information you have doesn't mention specific company names, you need to output a valid json with every piece that potentially partly answers the question:
#     {{
#         "relevant_context_pieces": [
#             {{"index": ..., "reason": ...,}},
#         ],
#     }}
# """
# #  "query": [brief and concise clarifying question to the vector datastore; avoid including information that the context piece potentially addresses; example: if the context piece give information about obligations, but no mention of the specififc company bearer of these obligations, you ask: "who is the contractor?"]    # You need to answer the question based on 1 context piece that fits the logical formula or several context pieces that collectively fit the logical formula (but only in case these several pieces can clearly be linked together; remember that these pieces may be from different agreements with different companies!)

#     # You will be given user's questions and a logical formulas that represent them.

# # It is very important that you know that different pices of context may come from different agreements with different companies. Unless the context explicitly mentions the company name requested by the user, you cannot assume that the context is related to that company.

# #     Answer the question based on the context.

# #     Avoid guessing, only use the context provided.



#     # that if 1 piece of information doesn't contain all the information for the answer and you cannot link 2 different pieces together by any identifyer (like company name, agreement number, etc), you cannot join them to answer the question.

# from langchain_core.prompts import PromptTemplate
# prompt = PromptTemplate.from_template(prompt_template)

# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4o", temperature=0)

# chain = prompt | llm

# answer = chain.invoke({
#     "question": simplified['question'],
#     "context": results_string
# })

# relevant_results = {}
# json_response = extract_fenced_text(answer.content)
# if json_response:
#     json_response = json.loads(json_response)
#     for res in json_response['relevant_context_pieces']:
#         if results[res['index']].metadata['agreement'] not in relevant_results:
#             relevant_results[results[res['index']].metadata['agreement']] = []
#         relevant_results[results[res['index']].metadata['agreement']].append(results[res['index']])
#     # print(f"Agreements: {relevant_results}")

#     for agreement, result in relevant_results.items():
#         print(f"Agreement: {agreement}")
#         vdb_contractor_results = vdb.similarity_search(
#             "what are the parties of this agreement?",
#             k=1,
#             filter={"agreement": agreement}
#         )
#         for res in vdb_contractor_results:
#             print(f"VDB contractor result: {res.page_content}\n")
#             relevant_results[agreement].append(res)
#         print("\n\n")

#         print(f"Relevant results: {relevant_results[agreement]}")
#         print(f"\n\n")

#         answer = chain.invoke({
#             "question": question,
#             "context": relevant_results[agreement]
#         })
#         print(f"Answer: {answer.content}")
#         print(f"\n\n")

#         json_response = extract_fenced_text(answer.content)
#         if not json_response:
#             break


#     # print(f"Relevant results: {relevant_results}")

#     print(f"\n")


# else:
#     print(answer.content)



# -------------------------------------------------------------------------------------------------
# Router

# from ai_agents import router

# chat_history = []

# while 1:
#     new_question = input("Enter a message: ")
#     if new_question == "exit":
#         break
#     response = router(HumanMessage(content=new_question))
#     print(response)


# -------------------------------------------------------------------------------------------------
# bemyapp_agent

from ai_agents import bemyapp_agent

current_agent = "text_to_sql_agent"
text2sql_chat_history = []

questions = {
    "text_to_sql_agent": [
        "What was the total profit for each product in the year 2014?",
        "how about 2013?",
        "by country",
    ],
    "switch_to_multimodal_rag_agent": [
        "I have a question about my video",
    ],
    "multimodal_rag_agent": [
        #  can be answered without images
        "What is the origin of Interactive Technical Manuals (ITMs), and how have they evolved since their creation?",
        "How can ITMs benefit technicians and engineers in troubleshooting and maintaining complex equipment?"
        # need images to answer
        "How does the ITM's schematic interface enhance the understanding of complex hydraulic systems?",
        "What kind of information is typically included in a component sheet within the ITM system?",
    ],
    "switch_to_recursive_rag_agent": [
        "I have a question about my contracts",
    ],
    "recursive_rag_agent": [
        "Which companies provide design services for us?",
        "What is the cost of services of CreativeMind?",
        "Give me contacts of CreativeMind",
    ]
}

while 1:
    question = input(f"Your message to {current_agent}: ")
    if question == "exit":
        break
    text2sql_chat_history.append(HumanMessage(content=question))
    response = bemyapp_agent(question, current_agent, text2sql_chat_history)
    current_agent = response["next_agent"]
    print(response["answer"])
    print("")
    print("")
