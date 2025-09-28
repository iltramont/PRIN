import os
from openai import OpenAI
from dotenv import load_dotenv  
from pprint import pprint

load_dotenv()  # Load environment variables from .env file

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

response = client.responses.create(
    model="gpt-4o-mini",
    input="Write a one-sentence bedtime story about a unicorn."
)

print("*--- Response from OpenAI:")
print(response.output_text)
print("*--- Resopnse:")
pprint(response)