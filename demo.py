from google import genai
from dotenv import load_dotenv
import os

# ✅ Load environment variables from .env file
load_dotenv()

# Now fetch the key
gemini_key = os.getenv("GEMINI_API_KEY")

# # Print to verify (temporary)
# print("Gemini Key Loaded:", gemini_key)

# Initialize the Gemini client
client = genai.Client(api_key=gemini_key)

resp = client.models.generate_content(
    model="gemini-1.5-flash",  # use available model name
    contents="Say hi"
)

print(resp.text)
