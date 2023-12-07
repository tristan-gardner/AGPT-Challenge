from typing import Callable, List
from constants import CHAR_LIMIT, DEFAULT_ANCIENT_SIZE, DEFAULT_PAST_SIZE, DEFAULT_RECENT_SIZE

summary_directive = "condense the information in this conversation - the response should be less than 1000 tokens and keep the dialog format of a user talking to an assistant meaning each line should start with 'user:' or 'assistant:'"
#"condense the information in this conversation - keep the format of a user talking to an assistant keep the response below 1000 tokens"

class WindowManager:
    # goal is to summarize the conversation with a bias towards recent messages
    def __init__(
            self, 
            oai_caller=Callable[[str, str, bool, str], str],
            rq_size=DEFAULT_RECENT_SIZE, 
            pq_size=DEFAULT_PAST_SIZE, 
            aq_size=DEFAULT_ANCIENT_SIZE,
        ) -> None:
        self.rq_size=rq_size
        self.pq_size=pq_size
        self.aq_size=aq_size
        self.oai_caller=oai_caller
        self.recent_queue: List[dict] = []
        self.past_queue: List[dict]  = []
        self.ancient_queue: List[dict]  = []
        self.context: List[dict]  = []

    def generate_summary(self, conversation: str) -> List[dict]:
        summary: str = self.oai_caller(
            system_prompt=summary_directive, 
            user_prompt=conversation,
        )
        lines = [line for line in summary.split("\n") if line.strip()]
        messages = []
        for line in lines:
            if line[:5].lower() == "user:":
                messages.append({"role": "user", "content": line[5:].strip()})
            elif line[:10].lower() == "assistant:":
                messages.append({"role": "assistant", "content": line[10:].strip()})
            else:
                if messages:
                    messages[-1]["content"] += f" {line.strip()}"
        
        return messages

    def get_context_window(self) -> List[dict]: 
        return self.context
    
    def handle_queue_updates(self, message: dict):
        # add message to the queues pushing older messages onto older queues
        self.recent_queue.append(message)
        if len(self.recent_queue) > self.rq_size:
            temp = self.recent_queue.pop(0)
            self.past_queue.append(temp)
        if len(self.past_queue) > self.pq_size:
            temp = self.past_queue.pop(0)
            self.ancient_queue.append(temp)
        if len(self.ancient_queue) > self.aq_size:
            self.ancient_queue.pop(0)

    def create_summary_for_queue(self, queue: List[dict]) -> List[dict]:
        # takes a queue which has user prompts and responses
        # separate the prompts and responses then summarize them
        conversation = ""
        for message in queue:
            if len(conversation) + len(message["content"]) > CHAR_LIMIT:
                break
            conversation += f"{message['role']}: {message['content']}\n"
        summary = self.generate_summary(conversation)
        return summary
    
    def add_message(self, message: dict) -> None:
        # adds a message to the queue and then updates the context window
        self.handle_queue_updates(message)
        past_summary, ancient_summary = [None], [None]

        if self.past_queue:
            past_summary = self.create_summary_for_queue(self.past_queue)

        if self.ancient_queue:
            ancient_summary = self.create_summary_for_queue(self.ancient_queue)
        
        context = [
            *ancient_summary,
            *past_summary,
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
# How important is the back and forth? If the context of the prompt is covered in the response do we need the prompt?