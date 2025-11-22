
import os
import time
from dotenv import load_dotenv
from cohere import ClientV2

# Load environment variables from .env file
load_dotenv()

def cohere_api():
    # Access API key
    cohere_api_key = os.getenv('COHERE_API_KEY')

    if not cohere_api_key:
        raise ValueError("COHERE API KEY not found, please check the .env file for key")

    co = ClientV2(api_key=cohere_api_key)
    print("API key loaded successfully from environment")
    return co
