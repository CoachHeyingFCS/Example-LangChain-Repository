from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun


model = ChatOllama(model="research-llm")
prompt = ChatPromptTemplate.from_template("""You are a helpful research AI. Use the following web search result: {context}. To write a concise, well-structured response to the question /"Who is : {question} /" """)
parser = StrOutputParser()
search = DuckDuckGoSearchRun()

chain = prompt | model | parser

def generate_response(userText):
    context = search.invoke(f"Who is {userText}")
    response = chain.invoke({"question": f"Summarize information about {userText}", "context": context})
    return response

def write_to_file(userText, information, i):
    file_name = "output.txt"
    try: 
        with open(file_name, 'a') as file:
            file.write(str(i)+") "+ userText + "\n")
            file.write(information)
            file.write("\n")
        print(f"Content sucessfully appended to {file_name}.")
    except IOError as e:
        print(f"An error occured: {e}")

count = 1
userInput = input("Write the name of the person you would like to learn about today below: \nType exit to leave\n\n")
while userInput.lower() != 'exit':
    aiResponse = generate_response(userInput)
    write_to_file(userInput, aiResponse, count)
    userInput = input("Who else do you want to learn about? *You can always type exit to leave*\n")
    count += 1
