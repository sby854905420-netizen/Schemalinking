import os
import json
import hashlib
from typing import List, Union
import openai
from transformers import pipeline
import torch
import os


try:
    import ollama
except ImportError:
    print("Please install libraries such as `ollama`, `transformers`, `openai`, etc!")

class LLM:

    def __init__(self, model_name: str = "llama3.2:1b", provider: str = "ollama", cache_dir: str = "cache"):
        """
        Initialize LLM processing class
        :param model_name: LLM model to use
        :param provider: Select LLM service provider ('openai', 'ollama', 'transformers')
        :param cache_dir: Cache directory
        """
        self.model_name = model_name
        self.provider = provider.lower()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _generate_cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    def _load_cache(self, key: str) -> Union[str, None]:
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f).get("response")
        return None

    def _save_cache(self, key: str, response: str):
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"response": response}, f, ensure_ascii=False)

    def _query_openai(self, prompt: str) -> str:
        """Call OpenAI API to generate text"""
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4096,
        )
        # print(response)
        return response.choices[0].message.content
    
    
    def _query_ollama(self, prompt: str) -> str:
        """Call local Ollama to generate text"""
        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]

    def _query_transformers(self, prompt: str) -> str:
        """Call local Hugging Face Transformers to generate text"""
        device = "cpu"
        if torch.cuda.is_available():
            device= torch.cuda.current_device()
        pipe = pipeline("text-generation", model="google/gemma-3-1b-it", device=device, torch_dtype="auto")
        messages = [
                    [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "You are a helpful assistant."},]
                        },
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt},]
                        },
                    ],
                ]
        response = pipe(messages, max_new_tokens=3000)
        output = response[0][0]['generated_text'][2]['content']
        return output

    def batch_query(self, prompts: List[str], use_cache: bool = True) -> List[str]:
        """
        Batch query multiple prompt words
        :param prompts: prompt word list
        :param use_cache: whether to use cache
        :return: generate text list
        """
        # return [self.query(prompt, use_cache=use_cache) for prompt in prompts]
        return [self.query(prompt) for prompt in prompts]


    def query(self, prompt: str):
        provider = self.provider
        if provider == "ollama":
            return self._query_ollama(prompt)
        elif provider == "openai":
            return self._query_openai(prompt)
        elif provider == "transformers":
            return self._query_transformers(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")




if __name__ == "__main__":

    llm = LLM(model_name = "gpt-5-nano", provider= "openai")

    query= "What is the capital of France?"
    query_list = ["What is the president of UK?","How long you have been alive?"]

    response = llm.query(query)
    print(f"Query: {query} \n {llm.model_name} Response: {response}")

    responses = llm.batch_query(query_list)

    for q,a in zip(query_list,responses):
        print(f"Query: {q} \n {llm.model_name} Response: {a}")
