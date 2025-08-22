import os
import time
import random
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing in .env file")

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# Create chat session with style instructions
chat = client.chats.create(
    model="gemini-1.5-flash",
    history=[
        {
            "role": "user",
            "parts": [
                {"text": (
                    "You are SoulScribe, a warm and empathetic journaling companion. "
                    "Always acknowledge the user's feelings first, reflect them back with understanding, "
                    "and then offer 1–2 gentle, supportive suggestions. "
                    "Keep responses under 4 sentences unless the user asks for more detail."
                )}
            ]
        },
        {
            "role": "model",
            "parts": [
                {"text": "Understood. I’ll keep my tone warm, empathetic, and concise."}
            ]
        }
    ]
)

def get_empathetic_response(user_input):
    for attempt in range(5):  # try up to 5 times
        try:
            response = chat.send_message(user_input)
            return response.text
        except Exception as e:
            if "UNAVAILABLE" in str(e) and attempt < 4:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"Server busy. Retrying in {wait:.1f} seconds...")
                time.sleep(wait)
            else:
                raise

if __name__ == "__main__":
    print("Welcome to SoulScribe. Share your thoughts, feelings, or experiences.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("SoulScribe: Take care. I'm here whenever you need to talk.")
            break
        empathetic_response = get_empathetic_response(user_input)
        print(f"SoulScribe: {empathetic_response}")
