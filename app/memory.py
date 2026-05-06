chat_history = {}

def save_message(session_id, role, content):
    if session_id not in chat_history:
        chat_history[session_id] = []
    chat_history[session_id].append({"role": role, "content": content})

def get_history(session_id):
    return chat_history.get(session_id, [])