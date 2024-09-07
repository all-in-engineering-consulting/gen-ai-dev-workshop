import re
import json
import os
import time

from typing import List, Union

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage
from langchain.schema import HumanMessage
from langchain.chains import create_sql_query_chain

from miltimodal_rag_preparation_functions import (
    extract_full_text_from_diarised_transcript,
    extract_only_word_ts_from_transcript,
    find_phrase_timestamps,
    process_segments,
    segments_to_langchain_documents,
)
from general_functions import extract_fenced_text, faiss_index


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def debug_info(message, to_print=True):
    if to_print:
        print(f"DEBUG: {message}\n")


debug_to_print = True


def recursive_rag_agent(question, vdb):

    # connect to the vector database

    vdb = vdb or faiss_index(index_name="faiss_index_recursive_rag")


    # search for relevant context pieces

    results = vdb.similarity_search(
        question,
    )
    results_string = "\n".join([f"Context piece {i}:\n{res.page_content}\n--\n" for i, res in enumerate(results)])


    simplify_question_prompt = SystemMessage(
        content="""
            Your role is to simplify user's questions and remove any company names or agreement numbers.

            Example:
            Input:
                "What are the NDA terms with RogaIKopyta?"
            Output (a valid json):
                {"question": "What are the NDA terms?", "company_name": "RogaIKopyta"}
        """
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    simplified_question_response = llm.invoke([
        simplify_question_prompt,
        HumanMessage(content=question)
    ])
    try:
        simplified = json.loads(extract_fenced_text(simplified_question_response.content))
    except:
        simplified = json.loads(simplified_question_response.content)

    debug_info(f"\n\n\n simplified question: {simplified} \n\n\n")


    prompt_template = """
        Your role is to answer user's questions related to legal agreements with their company (Innovate Spaces Corporation).

        Here's the user's question: {question}.

        If the company is not present in the user's query, consider it CreativeMind.
        
        Here are some context pieces from agreements with various companies:
        -----
        {context}
        -----

        In case you have all relevant info, answer the question in a concise manner.

        But if relevant information you have doesn't mention specific company names, you need to output a valid json with every piece that potentially partly answers the question:
        {{
            "relevant_context_pieces": [
                {{"index": ..., "reason": ...,}},
            ],
        }}
    """
    prompt = PromptTemplate.from_template(prompt_template)


    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm
    answer = chain.invoke({
        "question": simplified['question'],
        "context": results_string
    })

    relevant_results = {}
    json_response = extract_fenced_text(answer.content)
    try:
        json_response = json.loads(json_response)
    except:
        json_response = None

    if json_response:
        debug_info(f"...DEBUG: json_response: {json_response}")
        for res in json_response['relevant_context_pieces']:
            if results[res['index']].metadata['agreement'] not in relevant_results:
                relevant_results[results[res['index']].metadata['agreement']] = []
            relevant_results[results[res['index']].metadata['agreement']].append(results[res['index']])
        debug_info(f"Agreements: {relevant_results}")

        for agreement, result in relevant_results.items():
            debug_info(f"Agreement: {agreement}")
            vdb_contractor_results = vdb.similarity_search(
                "what are the parties of this agreement?",
                k=1,
                filter={"agreement": agreement}
            )
            for res in vdb_contractor_results:
                debug_info(f"VDB contractor result: {res.page_content}\n")
                relevant_results[agreement].append(res)
            # debug_info("\n\n")

            # debug_info(f"Relevant results: {relevant_results[agreement]}")
            # debug_info(f"\n\n")

            answer = chain.invoke({
                "question": question,
                "context": relevant_results[agreement]
            })
            debug_info(f"Answer: {answer.content}")
            debug_info(f"\n\n")

            json_response = extract_fenced_text(answer.content)
            try:
                json_response = json.loads(json_response)
            except:
                json_response = None
            if not json_response:
                return answer.content


        # debug_info(f"Relevant results: {relevant_results}")

        # debug_info(f"\n")

    else:
        return answer.content


def multimodal_rag_agent(question, vdb):

    vdb = vdb or faiss_index(index_name="faiss_index_multimodal_rag")

    results_media = vdb.similarity_search(
        question,
        filter={"type": "media"},
    )
    results_transcript = vdb.similarity_search(
        question,
        filter={"type": "transcript_chunk"},
    )

    results_media_string = "\n--\n\n".join([f"Context piece: {res.page_content} [{res.metadata}]" for res in results_media])
    results_transcript_string = "\n--\n\n".join([f"Context piece: {res.page_content} [{res.metadata}]" for res in results_transcript])

    prompt_template = """
        Here's the user's question: {question}.

        Here's the context that may be relevant:
        -----
        From video transcript:
        {results_transcript_string}
        -----
        From video media:
        {results_media_string}
        -----

        Ignore all the pieces of context that don't have information that adds up to the answer to the question.

        Answer the question based on the context.

        Avoid guessing, only use the context provided.

        If you used any video media context pieces to answer the question, please add "MEDIA: [value of 'media' field from the video media context piece that you used] (without quotes and square brackets)" at the end of your answer.
    """

    prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm
    answer = chain.invoke({
        "question": question,
        "results_transcript_string": results_transcript_string,
        "results_media_string": results_media_string,
    })

    return answer.content


def text_to_sql_results(question, db):

    db = db or SQLDatabase.from_uri("sqlite:///text_to_sql_example.db")

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

    start_time = time.time()
    # Execute the SQL query
    res = db.run(sql_query)
    execution_time = time.time() - start_time
    debug_info(f"SQL query execution time: {execution_time:.4f} seconds")

    return res, sql_query


def text_to_sql_agent(chat_history: List[Union[BaseMessage, SystemMessage]], db):

    if not isinstance(chat_history[-1], HumanMessage):
        debug_info("Last message must be a human message")
        return
    
    # Start tracking time for the entire function
    start_time = time.time()
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
    full_question = ai_response_full_question.content
    execution_time = time.time() - start_time
    debug_info(f"DEBUG: Full question creation execution time: {execution_time:.4f} seconds")

    start_time = time.time()
    sql_results_str, sql_query = text_to_sql_results(full_question, db)
    execution_time = time.time() - start_time
    debug_info(f"DEBUG: SQL query execution time: {execution_time:.4f} seconds")

    debug_info(f"DEBUG: sql_results_str: {sql_results_str}")

    start_time = time.time()
    # convert the sql results to human readabel answer using gpt-4o-mini
    system_prompt_sql_results = SystemMessage(content=f"""
        Here's the user's question: {full_question}.

        Here are the results of the SQL query:
        {sql_results_str}

        Convert the SQL query results to a human readable answer.

        Your answer must be concise and to the point.
    """)
    llm = ChatOpenAI(model="gpt-4o-mini")
    ai_response_sql_results = llm.invoke([system_prompt_sql_results])
    execution_time = time.time() - start_time
    debug_info(f"DEBUG: SQL results to human readable answer execution time: {execution_time:.4f} seconds")

    return ai_response_sql_results.content + f"\n\n[SQL query:\n{sql_query}]"


def router(latest_user_message: HumanMessage, current_agent: str):
    system_prompt = f"""
        Your role is to route the user to the correct agent.

        There are 3 agents available:
        1. Recursive RAG agent (for questions about agreements) (name for json: recursive_rag_agent)
        2. Multimodal RAG agent (for questions about the video) (name for json: multimodal_rag_agent)
        3. Text to SQL agent (for questions about the database of financial data) (name for json: text_to_sql_agent)

        Current agent the user is talking to: {current_agent}.

        You will get user's latest message and you need to identify if they want to switch from the current agent.

        You only switch if the user's message contains a clear intent to switch to a different agent.

        If they want to switch, you need to output a valid json with the name of the agent that should answer the question.

        If they don't want to switch, you need to output a valid json with "current" as the value for "agent" field.

        Example outputs:
        {{"agent": "recursive_rag_agent"}}
        {{"agent": "multimodal_rag_agent"}}
        {{"agent": "text_to_sql_agent"}}
        {{"agent": "current"}}
    """

    llm = ChatOpenAI(model="gpt-4o-mini")
    ai_response = llm.invoke([system_prompt, latest_user_message])

    try:
        ai_response_json = json.loads(extract_fenced_text(ai_response.content))
        return ai_response_json['agent']
    except:
        debug_info(f"ERROR")
        debug_info(f"DEBUG: ai_response: {ai_response}")
        debug_info(f"DEBUG: extract_fenced_text(ai_response): {extract_fenced_text(ai_response)}")
        return "current"


def bemyapp_agent(question, current_agent, text2sql_chat_history, vdb_multimodal, vdb_recursive, db):

    # debug_info(f"DEBUG: question: {question}")

    start_time = time.time()

    router_response = router(HumanMessage(content=question), current_agent)

    end_time = time.time()
    execution_time = end_time - start_time
    debug_info(f"DEBUG: Router execution time: {execution_time:.4f} seconds")

    # debug_info(f"DEBUG: router_response: {router_response}")
    # switch_to_agent = json.loads(extract_fenced_text(router_response))['agent']


    if router_response == "current":
    
    
        if current_agent == "text_to_sql_agent":
            start_time = time.time()
            answer = text_to_sql_agent(text2sql_chat_history, db)
            end_time = time.time()
            execution_time = end_time - start_time
            debug_info(f"DEBUG: Text to SQL agent execution time: {execution_time:.4f} seconds")
            return {"next_agent": current_agent, "answer": answer}
        elif current_agent == "multimodal_rag_agent":
            start_time = time.time()
            answer = multimodal_rag_agent(question, vdb_multimodal)
            end_time = time.time()
            execution_time = end_time - start_time
            debug_info(f"DEBUG: Multimodal RAG agent execution time: {execution_time:.4f} seconds")
            return {"next_agent": current_agent, "answer": answer}
        elif current_agent == "recursive_rag_agent":
            start_time = time.time()
            answer = recursive_rag_agent(question, vdb_recursive)
            end_time = time.time()
            execution_time = end_time - start_time
            debug_info(f"DEBUG: Recursive RAG agent execution time: {execution_time:.4f} seconds")
            return {"next_agent": current_agent, "answer": answer}
    else:
        current_agent = router_response
        return {"next_agent": current_agent, "answer": f"({current_agent}) Great! What is your question?"}

