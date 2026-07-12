from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


DTYPE_NAMES = {
    "bfloat16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}


class TransformersChatBackend:
    """Single-model, single-generation-lock Transformers chat backend."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        max_input_length: int = 24576,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        random_seed: int = 42,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Spider-Agent-TC Transformers inference requires torch and transformers."
            ) from exc

        if torch_dtype not in DTYPE_NAMES:
            raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device '{device}' requested, but CUDA is unavailable.")
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "--attn-implementation flash_attention_2 requires flash-attn. "
                    "Install a build compatible with the active CUDA/PyTorch stack or use sdpa."
                ) from exc

        self.torch = torch
        self.model_name = model_name
        self.device = device
        self.max_input_length = int(max_input_length)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.random_seed = int(random_seed)
        self.tools = tuple(deepcopy(dict(tool)) for tool in (tools or ()))
        self._generation_lock = threading.Lock()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = getattr(torch, DTYPE_NAMES[torch_dtype])
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map={"": device},
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _model_inputs(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        try:
            inputs = self.tokenizer.apply_chat_template(
                list(messages),
                tools=list(self.tools),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            input_ids = self.tokenizer.apply_chat_template(
                list(messages),
                tools=list(self.tools),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            inputs = {"input_ids": input_ids}

        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = self.torch.ones_like(inputs["input_ids"])
        return inputs

    def count_messages_tokens(self, messages: Sequence[Mapping[str, Any]]) -> int:
        token_ids = self.tokenizer.apply_chat_template(
            list(messages),
            tools=list(self.tools),
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(token_ids, "shape"):
            return int(token_ids.shape[-1])
        return len(token_ids)

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        generation_config: Mapping[str, Any] | None = None,
    ) -> str:
        inputs = self._model_inputs(messages)
        input_length = int(inputs["input_ids"].shape[-1])
        if input_length > self.max_input_length:
            raise ValueError(
                f"Chat input has {input_length} tokens, exceeding --max-input-length "
                f"{self.max_input_length}."
            )

        kwargs = deepcopy(dict(generation_config or {}))
        temperature = float(kwargs.pop("temperature", self.temperature))
        seed_offset = int(kwargs.pop("seed_offset", 0))
        kwargs.setdefault("max_new_tokens", self.max_new_tokens)
        kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        do_sample = bool(kwargs.pop("do_sample", temperature > 0))
        kwargs["do_sample"] = do_sample
        if do_sample:
            kwargs["temperature"] = temperature
        else:
            kwargs.pop("top_p", None)
            kwargs.pop("top_k", None)

        with self._generation_lock:
            generation_seed = self.random_seed + seed_offset
            self.torch.manual_seed(generation_seed)
            if self.device.startswith("cuda"):
                self.torch.cuda.manual_seed_all(generation_seed)
            with self.torch.inference_mode():
                outputs = self.model.generate(**inputs, use_cache=True, **kwargs)
        generated = outputs[0, input_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


__all__ = ["TransformersChatBackend"]
