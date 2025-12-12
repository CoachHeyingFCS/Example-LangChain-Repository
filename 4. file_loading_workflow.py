from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_community.document_loaders import PyPDFLoader

#--Important Variables--#
#The location of the PDF in my repository
PDF_PATH = "/workspaces/Example-LangChain-Repository/Understanding Modelfile in Ollama.pdf"
#The name of the LLM I made when I did ollama create test-llm -f Modelfile
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
    #Splits the DPF into indivudal pages based around document objects
    document = loader.load_and_split()
    #Returns a list of Document Objects
    #Document objects are a tuple with the first part being the page content or the words on the page
    #The second part is a dictionary of metadata (source and page number)
    #Example:
    #[
    #    Document(page_content="Text from page 1", metadata={"source": "file.pdf", "page": 1}),
    #    Document(page_content="Text from page 2", metadata={"source": "file.pdf", "page": 2}),
    #    Document(page_content="Text from page 3", metadata={"source": "file.pdf", "page": 3})   
    #]
    return document

#Gives a string version of the whole content of the Document Object passed as the parameter sections
def format_documents(sections):
    outputString = ""
    #Goes through every element of the list sections
    for s in sections:
        #pulls out the page content from the tuple for each document object
        outputString += s.page_content
    return outputString

def ask(query):
    #Create the list of document objects
    pdf = loadPDF()
    #Pulls the text out of all the document objects
    pdfText = format_documents(pdf)
    #Runs the chain with the query and the text as context
    result = chain.invoke({"input": query,"context":pdfText}) 
    print(result)

user_input = input("What is your question?\n\n")
while user_input.lower() != 'exit':
    ask(user_input)
    user_input = input("What is your question?\n\n")
