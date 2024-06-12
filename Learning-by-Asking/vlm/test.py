# from openai import OpenAI
import os
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


# MODEL = "gpt-3.5-turbo"
# USER_INPUT_MSG = "chatGPT에 대해 설명해줘."

# response = openai.ChatCompletion.create(
#     model=MODEL,
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": USER_INPUT_MSG}, 
#         {"role": "assistant", "content": "Who's there?"},
#     ],
#     temperature=0,
# )

# print(response.choices[0].message.content)