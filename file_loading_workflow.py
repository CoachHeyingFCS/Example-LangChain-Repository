from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 

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
    #Gets the format in a document file
    document = loader.load_and_split()
    #Stores all of the text in a single string
    full_text = ""
    for page in document:
        thisPage = page.page_content
        full_text += thisPage
    
    return thisPage




def ask(query):
    pdfText = loadPDF()
    # invoke chain
    result = chain.invoke({"input": query,"context":pdfText})  # print results
    print(result)#["answer"])


user_input = input("What is your question?\n\n")
while user_input.lower() != 'exit':
    ask(user_input)
    user_input = input("What is your question?\n\n")
