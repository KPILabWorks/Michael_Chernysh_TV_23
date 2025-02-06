from dotenv import load_dotenv
import os

load_dotenv()

TG_API_KEY = os.getenv("TG_API_KEY")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
