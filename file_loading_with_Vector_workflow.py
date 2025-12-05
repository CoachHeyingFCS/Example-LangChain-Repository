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

#Setting up variables in my chain
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

#Invoking my chain
chain = prompt | model | parser

def loadPDF():
    # Pull PDF into code
    loader = PyPDFLoader(PDF_PATH)
    #Split PDF into individual pages
    pages = loader.load_and_split()
    #Creates the tool that will split the pages into individual chunks based on the size of chunk you want
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap = 200,
        length_function = len,
        add_start_index = True
    )
    #Does the splitting
    chunks = text_splitter.split_documents(pages)
    print(f"Split {len(pages)} pages into {len(chunks)} chunks")
    return chunks
    
def create_vectors(chunks):
    #Create vector database for pdf so it is easier for LLM to read
    embedding = FastEmbedEmbeddings()
    #Stores the vector database in a Chroma database (also stored in the repository.)
    database = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=DB_DIR)
    #returns the Chroma database of vectors
    return database

def format_documents(sections):
    outputString = ""
    for s in sections:
        outputString += s.page_content
    return outputString

def ask(query):
    #Load the Documents and chunk them
    document_chunks = loadPDF()
    #Create the vectors and store them in a Chroma database
    vectors = create_vectors(document_chunks)
    #Retreive the vectors that are relevant
    retriever = vectors.as_retriever(
        search_type = "similarity_score_threshold",
        search_kwargs={
            "k": 2,
            #This is the threashold for how relevant a chunk should be to be considered helpful
            "score_threshold": 0.5,
        }
    )
    #Get the relevant chunks of document from the vectors
    document_chain = retriever.invoke(query)
    #make the relevant chunks of documents into a string
    context = format_documents(document_chain)
    # invoke chain with the string of the query and the context for the prompt
    result = chain.invoke({"input": query, "context":context}) 
    # print results
    print(result)


user_input = input("What is your question?\n\n")
while user_input.lower() != 'exit':
    ask(user_input)
    user_input = input("What is your question?\n\n")
