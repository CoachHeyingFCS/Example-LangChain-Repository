from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_community.document_loaders import PyPDFLoader

#--Important Variables--#
PDF_PATH = "/workspaces/Example-LangChain-Repository/Understanding Modelfile in Ollama.pdf"
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
    return document

def format_documents(sections):
    outputString = ""
    for s in sections:
        outputString += s.page_content
    return outputString

def ask(query):
    pdf = loadPDF()
    pdfText = format_documents(pdf)
    result = chain.invoke({"input": query,"context":pdfText}) 
    print(result)

user_input = input("What is your question?\n\n")
while user_input.lower() != 'exit':
    ask(user_input)
    user_input = input("What is your question?\n\n")
