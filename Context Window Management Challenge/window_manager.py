from constants import CHARS_PER_TOKEN, TOKEN_LIMIT
import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import Text8Corpus
from gensim.similarities.annoy import AnnoyIndexer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


## ideas
# use gensim to find the most similar lines in the history - use 4k tokens from the most similar lines
class WindowManager:
    def __init__(self) -> None:
        self.model = Doc2Vec(vector_size=50, min_count=2, epochs=20)
        self.history = []

    def add_to_history(self, message:str ):
        self.history.append(message)
        tagged_prompt = TaggedDocument(words=message.split(" "), tags=[str(len(self.history))])
        self.model.build_vocab([tagged_prompt], update=True)
        self.model.train([tagged_prompt], total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def get_most_similar(self, message:str):
        new_vector = self.model.infer_vector(message.split(" "))
        similar_prompts = self.model.docvecs.most_similar([new_vector], topn=min(20, self.model.corpus_count))
        similar_prompt_indices = [int(index) for index, _ in similar_prompts]
        most_similar_prompts = [self.history[index] for index in similar_prompt_indices]
        return most_similar_prompts
    
    def get_context_window(self, prompt: str):
        window = []
        most_similar_prompts = self.get_most_similar(prompt)
        most_recent_prompts = most_similar_prompts[-5:]
        # use up to 1/4 of tokens on recent prompts
        recent_tokens = TOKEN_LIMIT // 4
        for prompt in most_recent_prompts:
            prompt_tokens = len(prompt) // CHARS_PER_TOKEN
            if prompt_tokens < recent_tokens:
                window.append(prompt)
                recent_tokens -= prompt_tokens
            else:
                break

        # use up to 3/4 of tokens on similar prompts
        similar_tokens = TOKEN_LIMIT - recent_tokens
        for prompt in most_similar_prompts:
            prompt_tokens = len(prompt) // CHARS_PER_TOKEN
            if prompt_tokens < similar_tokens:
                window.append(prompt)
                similar_tokens -= prompt_tokens
            else:
                break

        return window

