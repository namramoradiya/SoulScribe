# import os
# import time
# import random
# from dotenv import load_dotenv
# from google import genai

# # Load environment variables from .env
# load_dotenv()

# # Get the API key from the environment
# api_key = os.getenv("GEMINI_API_KEY")

# if not api_key:
#     raise ValueError("GEMINI_API_KEY is missing in .env file")

# # Initialize Gemini client
# client = genai.Client(api_key=api_key)

# # Create chat session with style instructions
# chat = client.chats.create(
#     model="gemini-1.5-flash",
#     history=[
#         {
#             "role": "user",
#             "parts": [
#                 {"text": (
#                     "You are SoulScribe, a warm and empathetic journaling companion. "
#                     "Always acknowledge the user's feelings first, reflect them back with understanding, "
#                     "and then offer 1–2 gentle, supportive suggestions. "
#                     "Keep responses under 4 sentences unless the user asks for more detail."
#                 )}
#             ]
#         },
#         {
#             "role": "model",
#             "parts": [
#                 {"text": "Understood. I’ll keep my tone warm, empathetic, and concise."}
#             ]
#         }
#     ]
# )

# def get_empathetic_response(user_input):
#     for attempt in range(5):  # try up to 5 times
#         try:
#             response = chat.send_message(user_input)
#             return response.text
#         except Exception as e:
#             if "UNAVAILABLE" in str(e) and attempt < 4:
#                 wait = (2 ** attempt) + random.uniform(0, 1)
#                 print(f"Server busy. Retrying in {wait:.1f} seconds...")
#                 time.sleep(wait)
#             else:
#                 raise

# if __name__ == "__main__":
#     print("Welcome to SoulScribe. Share your thoughts, feelings, or experiences.")
#     while True:
#         user_input = input("\nYou: ")
#         if user_input.lower() in ["exit", "quit", "bye"]:
#             print("SoulScribe: Take care. I'm here whenever you need to talk.")
#             break
#         empathetic_response = get_empathetic_response(user_input)
#         print(f"SoulScribe: {empathetic_response}")



import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the client
genai.configure(api_key=API_KEY)

# --- Step 1: List available models to confirm ---
print("Available models:")
for m in genai.list_models():
    print(m.name)

# --- Step 2: Choose a valid model from the list ---
# Replace this with any supported chat model you see in the list
MODEL_NAME = "models/gemini-2.5-flash"  # Example: change if your list shows something else
model = genai.GenerativeModel(MODEL_NAME)

# --- Step 3: Start a chat session ---
chat = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                "You are SoulScribe, a warm and empathetic journaling companion. "
                "Always acknowledge the user's feelings first, reflect them back, "
                "and offer 1–2 gentle, supportive suggestions. Keep responses under 4 sentences."
            ]
        },
        {
            "role": "model",
            "parts": [
                "Understood. I’ll keep my tone warm, empathetic, and concise."
            ]
        }
    ]
)

# --- Step 4: Example of sending a message ---
def get_soulscribe_response(user_input):
    """
    Sends user input to the AI and returns SoulScribe's response.
    """
    response = chat.send_message(user_input)
    return response.text

# Example usage
if __name__ == "__main__":
    user_text = "I'm feeling a bit stressed today."
    reply = get_soulscribe_response(user_text)
    print("SoulScribe:", reply)
