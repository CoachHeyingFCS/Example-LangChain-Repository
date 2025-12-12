from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="chatbob")
prompt = ChatPromptTemplate.from_template("""{question}""")
parser = StrOutputParser()

chain = prompt | model | parser

userInput = input("Write what you would like to say to SpongeBob below: \nType exit to leave\n\n")
while userInput.lower() != 'exit':
    response = chain.invoke({"question": userInput})
    print(response)
    userInput = input("*You can always type exit to leave*\n")

print("Good-bye. Thank you for using my program")