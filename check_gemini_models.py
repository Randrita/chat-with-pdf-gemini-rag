from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

models = client.models.list()

for model in models:
    print("Model name:", model.name)
    print("  Display name:", model.display_name)
    print("  Description:", model.description)
    print("  Input token limit:", model.input_token_limit)
    print("  Output token limit:", model.output_token_limit)
    print("-" * 60)
