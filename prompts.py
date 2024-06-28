from langchain.prompts import PromptTemplate

description_prompt = PromptTemplate.from_template("Write me a description for a Tiktok about {topic}")

script_prompt = PromptTemplate.from_template("Write me a script for a Tiktok given the following  descritption: {description}")