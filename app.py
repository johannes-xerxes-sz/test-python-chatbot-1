from flask import Flask, request, jsonify, render_template
from chatbot import ChatBot

app = Flask(__name__)
chatbot = ChatBot()

# Route to serve the homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle chat messages
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    print(f"Received message: {user_message}")  # Debugging

    if not user_message:
        return jsonify({"response": "Please type a message.", "sentiment": "Neutral"})

    # Get the chatbot response and sentiment
    bot_response, user_sentiment = chatbot.get_response(user_message)
    print(f"Bot response: {bot_response}, Sentiment: {user_sentiment}")  # Debugging

    return jsonify({"response": bot_response, "sentiment": user_sentiment})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
