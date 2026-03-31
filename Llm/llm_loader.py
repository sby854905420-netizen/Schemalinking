import os
from typing import List, Union
import openai
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Mistral3ForConditionalGeneration, FineGrainedFP8Config, MistralCommonBackend
import torch
import ollama

from config import *


class LLM:

    def __init__(
        self,
        model_name: str = None,
        provider: str = None,
        num_ctx: Union[int, None] = None,
        think_mode: bool = False,
        query_settings: dict = None,
    ):
        """
        Initialize LLM processing class
        :param model_name: LLM model to use
        :param provider: Select LLM service provider ('openai', 'ollama', 'transformers')
        """
        if model_name is None:
            self.model_name = ANSWER_LLM_NAME
        else:
            self.model_name = model_name

        if provider is None:
            self.provider = PROVIDER
        else:
            self.provider = provider.lower()

        if num_ctx is None:
            self.num_ctx = CONTEXT_WINDOW_SIZE
        else:
            self.num_ctx = num_ctx

        self.think_mode = think_mode

        if query_settings is None:
            self.query_settings = LLM_SETTINGS
        else:
            self.query_settings = query_settings

        if torch.cuda.is_available():
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        self.device_map = "auto"

        if self.provider == "transformers":
            if "ministral" in self.model_name.lower():
                self._load_ministral_model()
            else:
                self._load_transformers_model()
        
        

    def _load_ministral_model(self):
        self.tokenizer = MistralCommonBackend.from_pretrained(self.model_name)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            quantization_config=FineGrainedFP8Config(dequantize=True),
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _query_ministral(self, prompt: str,output_hidden_states:bool=False):
        messages = [{"role": "user", "content": prompt}]
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(self.model.device)    
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        generation_kwargs = dict(self.query_settings)
        generation_kwargs.setdefault("do_sample", generation_kwargs.get("temperature", 0.0) > 0)
        generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        generation_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

        if output_hidden_states:
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
            return outputs.logits[:, -1, :].float()
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.num_ctx,
                    **generation_kwargs,
                )
            generated_tokens = outputs[0][input_ids.shape[-1]:]
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


    def _load_transformers_model(self):
        """Load local or Hugging Face causal language model"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _query_transformers(self, prompt: str,output_hidden_states:bool=False):
        """Call local transformers model to generate text"""
        messages = [{"role": "user", "content": prompt}]

        if hasattr(self.tokenizer, "apply_chat_template"):
            model_inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            formatted_prompt = prompt
            model_inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                max_length=self.num_ctx,
            )

        if isinstance(model_inputs, torch.Tensor):
            input_ids = model_inputs.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)
        else:
            model_inputs = model_inputs.to(self.model.device)
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

        generation_kwargs = dict(self.query_settings)
        generation_kwargs.setdefault("do_sample", generation_kwargs.get("temperature", 0.0) > 0)
        generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        generation_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

    
        if output_hidden_states:
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
            return outputs.logits[:, -1, :].float()
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.num_ctx,
                    **generation_kwargs,
                )
            generated_tokens = outputs[0][input_ids.shape[-1]:]
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    

    
    def _query_openai(self, prompt: str) -> str:
        """Call OpenAI API to generate text"""
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        # print(response)
        return response.choices[0].message.content
    
    def _query_ollama(self, prompt: str) -> str:
        """Call local Ollama to generate text"""
        LLM_SETTINGS['num_ctx'] = self.num_ctx
        response = ollama.chat(model=self.model_name, 
                               messages=[{"role": "user", "content": prompt}],
                               format="json",
                               think=THINKING_MODE,
                               options=LLM_SETTINGS
                               )
        return response["message"]["content"]

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
            if "ministral" in self.model_name.lower():
                return self._query_ministral(prompt)
            return self._query_transformers(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")




if __name__ == "__main__":

    llm = LLM()
    print(f"Initilize the {llm.model_name} on {llm.device_map}, provider is {llm.provider}")
    query= "What is the capital of France?"
    query_list = ["What is the president of UK?","How long you have been alive?"]

    response = llm.query(query)
    print(f"Query: {query} \n {llm.model_name} Response: {response}")

    # responses = llm.batch_query(query_list)

    # for q,a in zip(query_list,responses):
    #     print(f"Query: {q} \n {llm.model_name} Response: {a}")
