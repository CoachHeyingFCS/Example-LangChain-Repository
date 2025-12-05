#IMPORTANT: This solution is not great and sometimes I had to ask the same question multiple times.
#I prioritized reability and making it similar to the other code in this library and lost some acccuracy/speed
#Feel free to explore some other options for creating the chain as this one doesn't use the vectors to the best of their ability

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma


#--Important Variables--#
PDF_PATH = "/workspaces/Example-LangChain-Repository/Understanding Modelfile in Ollama.pdf"
DB_DIR = "./sql_chroma_db"
MODEL_NAME = "test-llm"

model = ChatOllama(model= MODEL_NAME)
prompt = ChatPromptTemplate.from_template(
    """ 
    You are an AI assistant. Use ONLY the provided context to answer the question. 
    If the answer is not clearly and directly supported by the context, respond exactly with:
    "I don't have enough context to answer that."

    Do NOT make up facts or speculate.
    "You must base your answer ONLY on the provided context. 
    If you include any information from the context, you must reference the filename or page it came from. 
    If there is no relevant context, respond: 'I don't have enough context to answer that.'"

    Question:
    {input}

    Context:
    {context}

    Answer:
    """
)
parser = StrOutputParser()

chain = prompt | model | parser

def loadPDF():
    # Pull PDF into code
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap = 200,
        length_function = len,
        add_start_index = True
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Split {len(pages)} pages into {len(chunks)} chunks")

    embedding = FastEmbedEmbeddings()
    database = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=DB_DIR)
    return database

def format_documents(sections):
    outputString = ""
    for s in sections:
        outputString += s.page_content
    return outputString

def ask(query):
    vectors = loadPDF()
    retriever = vectors.as_retriever(
        search_type = "similarity_score_threshold",
        search_kwargs={
            "k": 2,
            "score_threshold": 0.5,
        }
    )
    document_chain = retriever.invoke(query)
    context = format_documents(document_chain)
    # invoke chain
    result = chain.invoke({"input": query, "context":context}) 
    # print results
    print(result)


user_input = input("What is your question?\n\n")
while user_input.lower() != 'exit':
    ask(user_input)
    user_input = input("What is your question?\n\n")