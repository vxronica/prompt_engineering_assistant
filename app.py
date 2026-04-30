from flask import Flask, render_template, request, jsonify
from chatbot import ask_chatbot

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"response": "Please enter a question."})

    bot_response = ask_chatbot(user_message)

    return jsonify({"response": bot_response})


if __name__ == "__main__":
    app.run(debug=True, port=5050)