# from flask import Flask, render_template, request, jsonify
# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI

# app = Flask(__name__)

# load_dotenv()

# # Load the Google API key from environment variables
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # Initialize the Gemini model
# llm4 = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# # Define your query
# # while True:
# #     text = input()
# #     response = llm4.predict(text)
# #     print(response)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/query', methods = ['POST'])
# def query():
#     user_input = request.json.get('text')
#     response = llm4.predict(user_input)
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)




####################################################################################################################
# import os

# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain, SimpleSequentialChain

# from prompts import script_prompt, description_prompt

# # vector store implementation
# from langchain_core.documents import Document
# from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# # tweet_prompt = PromptTemplate.from_template("I want to open a restaurant for {cuisine} food suggest a fancy name for it.")
# # tweet_prompt = PromptTemplate.from_template("You are a content creator. Write me a tweet about {topic}.")

# # tweet_chain = LLMChain(llm=llm, prompt=tweet_prompt, verbose=True)

# # if __name__=="__main__":
# # while True:
# #     cuisine = input()
# #     resp = tweet_chain.run(cuisine)
# #     print(resp)
    

# description_chain = LLMChain(llm=llm, prompt=description_prompt)
# # output = description_chain.predict(topic="quantom physics is awesome")
# # print(output)

# script_chain = LLMChain(llm=llm, prompt=script_prompt)
# # script = script_chain.predict(description=output, verbose=True)
# # print(script)

# tiktok_chain = SimpleSequentialChain(chains=[description_chain, script_chain], verbose=True)
# script = tiktok_chain.run("quantum physics is awesome")

# print(script)

# # vector  store and retrever implementation
# documents = [
#     Document(
#         page_content="Dogs are great companions, known for their loyalty and friendliness.",
#         metadata={"source": "mammal-pets-doc"},
#     ),
#     Document(
#         page_content="Cats are independent pets that often enjoy their own space.",
#         metadata={"source": "mammal-pets-doc"},
#     ),
#     Document(
#         page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
#         metadata={"source": "fish-pets-doc"},
#     ),
#     Document(
#         page_content="Parrots are intelligent birds capable of mimicking human speech.",
#         metadata={"source": "bird-pets-doc"},
#     ),
#     Document(
#         page_content="Rabbits are social animals that need plenty of space to hop around.",
#         metadata={"source": "mammal-pets-doc"},
#     ),
# ]

# # embeddings = GoogleGenerativeAIEmbeddings(model="google_gemini", google_api_key=GOOGLE_API_KEY)

# vectorstor = Chroma.from_documents(
#     documents,
#     embedding=GoogleGenerativeAIEmbeddings(model="gemini-pro", google_api_key=GOOGLE_API_KEY),
#     # embedding=embeddings,
# )

# result = vectorstor.similarity_search("cat")

# # for result in result:
# #     print(f"Document ID: {result.metadata['source']}, Content: {result.page_content}")

#######################################################################################################################







import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_core.documents import Document
from langchain_chroma import Chroma
from typing import List
from langchain_core.runnables import RunnableLambda
from prompts import script_prompt, description_prompt

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=GOOGLE_API_KEY)

# description_chain = LLMChain(llm=llm, prompt=description_prompt)
# script_chain = LLMChain(llm=llm, prompt=script_prompt)

# tiktok_chain = SimpleSequentialChain(chains=[description_chain, script_chain], verbose=True)

# script = tiktok_chain.run("quantum physics is awesome")

# print(script)

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_KEY=GOOGLE_API_KEY)

vectorstore = Chroma.from_documents(documents, embedding=embeddings)

# result = vectorstore.similarity_search("cat")

# embedding = embeddings.embed_query("cat")
# result = vectorstore.similarity_search_by_vector(embedding)
# # Print search results (uncomment to see output)
# for result in result:
#     print(f"Document ID: {result.metadata['source']}, Content: {result.page_content}")



retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)

result = retriever.batch(["cat", "shark"])

for docs in result:
    for doc in docs:
        print(f"Document ID: {doc.metadata['source']}, Content: {doc.page_content}")
