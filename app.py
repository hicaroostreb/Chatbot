from flask import Flask, request, jsonify

from bot.ai_bot import AIBot
from services.waha import Waha

app = Flask(__name__)


@app.route("/chatbot/webhook/", methods=["POST"])
def webhook():
    data = request.json
    chat_id = data["payload"]["from"]
    received_message = data["payload"]["body"]
    is_group = "@g.us" in chat_id
    is_status = "status@broadcast" in chat_id

    if is_group or is_status:
        return jsonify(
            {"status": "ok", "message": "Mensagem de grupo/status ignorada."}
        ), 200

    waha = Waha()
    ai_bot = AIBot()

    waha.start_typing(chat_id=chat_id)
    history_messages = waha.get_history_messages(
        chat_id=chat_id,
        limit=5,
    )
    response_message = ai_bot.invoke(
        history_messages=history_messages,
        question=received_message,
    )
    waha.send_message(
        chat_id=chat_id,
        message=response_message,
    )
    waha.stop_typing(chat_id=chat_id)

    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
