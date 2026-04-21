import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

def get_ai_response(user_message):
    prompt = f"""
    You are an intelligent agriculture assistant.

    Give short, practical advice for farmers.

    Help with:
    - irrigation decisions
    - crop recommendations
    - fertilizer usage
    - soil health
    - weather-based suggestions

    User: {user_message}
    """

    response = model.generate_content(prompt)
    return response.text