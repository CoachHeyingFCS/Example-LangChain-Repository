import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

model = ChatOllama(model="web-llm", temperature=0)
prompt = ChatPromptTemplate.from_template(
    "Summarize the following content fom a webpage:\n"
    "Content:\n{content}"
)
parser = StrOutputParser()

# Create the chain
chain = prompt | model | parser

# Loads a webpage, chunk the text for long pages, and generate a structured summary.
def summarize_url(url: str):
    try:
        website = WebBaseLoader(url)
        pageText = website.load()

        if not pageText:
            return "No content found."
    except Exception as e:
        return f"Failed to load URL: {e}"

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = splitter.split_documents(pageText)
    # Combine chunks
    text = ""
    for chunk in chunks:
        text += chunk.page_content
    # Run the chain
    return chain.invoke({"content": text})


url = input("What website would you like to summarize?\n")
print("\n=== SUMMARY ===\n")
print(summarize_url(url))

