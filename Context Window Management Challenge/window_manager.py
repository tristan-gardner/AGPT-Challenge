from constants import CHARS_PER_TOKEN, TOKEN_LIMIT


token_buffer = 100

class WindowManager:
    def __init__(self) -> None:
        self.full_message_history = []
        self.token_count = 0
        self.start_index = 0

    def get_context_window(self): 
        return self.full_message_history[self.start_index:]
    
    def add_message(self, message: dict):
        self.full_message_history.append(message)
        self.token_count += len(message['content'])//CHARS_PER_TOKEN
        while self.token_count >= TOKEN_LIMIT - token_buffer and self.start_index < len(self.full_message_history):
            self.token_count -= len(self.full_message_history[self.start_index]['content'])//CHARS_PER_TOKEN
            self.start_index += 1