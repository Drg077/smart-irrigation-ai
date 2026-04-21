from flask import Blueprint, request, jsonify
from services.chatbot_services import get_ai_response
from flask_cors import CORS
chatbot_bp = Blueprint("chatbot", __name__)
CORS(chatbot_bp)  # Allow CORS for the React frontend
@chatbot_bp.route("/chat", methods=["POST"])
def chat():
    
    data = request.json
    print(data)
    user_message = data.get("message")

    if not user_message:
        return jsonify({"response": "Please provide a message"}), 400

    response = get_ai_response(user_message)

    return jsonify({"response": response})