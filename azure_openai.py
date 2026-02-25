# Import Azure OpenAI
# from langchain_openai import AzureOpenAI
import os
from dotenv import load_dotenv
# from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

load_dotenv()

# try:
#     print(os.getenv("AZURE_INFERENCE_ENDPOINT"))
#     print(os.getenv("AZURE_INFERENCE_CREDENTIAL"))
# except Exception as e:
#     print(e)
import os
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

model = AzureAIChatCompletionsModel(
    endpoint=os.environ["AZURE_INFERENCE_ENDPOINT"],
    credential=os.environ["AZURE_INFERENCE_CREDENTIAL"],
    model="gpt-4o",
)
response = model.invoke("Say hello")
print(response.content)