import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION')
)

response = client.chat.completions.create(
    model=os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT'),
    messages=[{'role': 'user', 'content': 'Say: Azure OpenAI connected successfully!'}],
    max_tokens=20
)

print(response.choices[0].message.content)