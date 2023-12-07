class WindowManager:
    def __init__(self) -> None:
        self.full_message_history = []

    def get_context_window(self): 
        # change this
        return self.full_message_history[-4:]
    
    def add_message(self, message):
        self.full_message_history.append(message)