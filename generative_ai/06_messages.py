from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

from langchain.messages import SystemMessage, HumanMessage

model = init_chat_model("gpt-4o-mini")

system_msg = SystemMessage("""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
""")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)
# print(response.usage_metadata)
print(response["messages"])




# model = init_chat_model("gpt-4o-mini")

# Pass Human Message to the model
# response = model.invoke([
#   HumanMessage("What is machine learning?")
# ])
# print(response)








# response = model.invoke([
#   SystemMessage(content="You are a helpful assistant that can answer questions and help with tasks."),
#   HumanMessage(content="What is the capital of France?"),
#   AIMessage(content="The capital of France is Paris.")
# ])
# print(response)

# messages = [
#   SystemMessage(content="You are a helpful assistant that can answer questions and help with tasks."),
#   HumanMessage(content="What is the capital of France?"),
#   AIMessage(content="The capital of France is Paris.")
  
# ]

# print(messages[2].tool_calls)
