from typing import List
from summarizer import Summarizer
from constants import CHARS_PER_TOKEN, TOKEN_LIMIT

CHAR_LIMIT = TOKEN_LIMIT * CHARS_PER_TOKEN
RECENT_SIZE = 4
PAST_SIZE = 8
ANCIENT_SIZE = 16
# Create a Summarizer object
summarizer = Summarizer()


def generate_summary(text, max_length=CHAR_LIMIT, min_length=62):
    if len(text) < min_length:
        return text
    # Generate summary
    summary = summarizer(text, max_length=max_length)

    return summary


class WindowManager:
    # goal is to summarize the conversation with a bias towards recent messages
    def __init__(self) -> None:
        self.full_message_history = []
        self.recent_queue: List[dict] = []
        self.past_queue: List[dict]  = []
        self.ancient_queue: List[dict]  = []
        self.context: List[dict]  = []

    def get_context_window(self) -> List[dict]: 
        # change this
        return self.context
    
    def handle_queue_updates(self, message: dict):
        self.full_message_history.append(message)
        #push older messages into older queues
        self.recent_queue.append(message)
        if len(self.recent_queue) > RECENT_SIZE:
            self.recent_queue.pop()
            self.past_queue.append(message)
        if len(self.past_queue) > PAST_SIZE:
            self.past_queue.pop()
            self.ancient_queue.append(message)
        if len(self.ancient_queue) > ANCIENT_SIZE:
            self.ancient_queue.pop()

    def create_summary_for_queue(self, queue: List[dict], char_limit: int) -> List[dict]:
        user_prompts = [message["content"] for message in queue if message['role'] == 'user']
        responses = [message["content"] for message in queue if message['role'] == 'assistant']
        # create a summary of the user prompts
        user_prompt_text = " ".join([content for content in user_prompts])
        user_summary = generate_summary(user_prompt_text)

        # create summary for responses
        response_text = " ".join([content for content in responses])
        response_summary = generate_summary(response_text, max_length=char_limit)

        return [
            {"role": "user", "content": user_summary},
            {"role": "assistant", "content": response_summary}
        ]
    
    def add_message(self, message: dict):
        self.handle_queue_updates(message)
        chars_to_use = CHAR_LIMIT - sum([len(message["content"]) for message in self.recent_queue])
        past_user_summary, past_response_summary, ancient_user_summary, ancient_response_summary = None, None, None, None

        if self.past_queue:
            past_user_summary, past_response_summary = self.create_summary_for_queue(self.past_queue, chars_to_use/2)

        if self.ancient_queue:
            ancient_user_summary, ancient_response_summary = self.create_summary_for_queue(self.ancient_queue, chars_to_use/2)
        
        context = [
            ancient_user_summary,
            ancient_response_summary,
            past_user_summary,
            past_response_summary,
            *self.recent_queue,
        ]

        self.context = [message for message in context if message is not None]

# Some Open questions
# how good is the summarizer?
# If we summarize a list of question and answers does it make sense to the llm the same way a full list would?
# what is the right size of each queue?
# what is the right amount of queues?
# when if ever can we forget a message?
# Am I right to give preference to recent messages?