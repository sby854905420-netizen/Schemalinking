import os
from typing import List, Sequence, Union
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
from config import *

class EmbeddingModelLoader:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        cache_dir: str = PROJECT_ROOT / "Llm" / "cache",
        device: Union[str, None] = None,
        normalize_embeddings: bool = True,
        trust_remote_code: bool = True,
        backend_preference: Sequence[str] = ("sentence_transformers", "transformers"),
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.normalize_embeddings = normalize_embeddings
        self.trust_remote_code = trust_remote_code
        self.backend_preference = list(backend_preference)
        os.makedirs(cache_dir, exist_ok=True)

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.backend = None
        self.model = None
        self.tokenizer = None
        self._load_model()


    def _load_with_sentence_transformers(self) -> bool:
        if SentenceTransformer is None:
            return False

        self.model = SentenceTransformer(
            self.model_name,
            cache_folder=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
            device=self.device,
        )
        self.backend = "sentence_transformers"
        return True

    def _load_with_transformers(self) -> bool:
        if AutoTokenizer is None or AutoModel is None:
            return False

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()
        self.backend = "transformers"
        return True

    def _load_model(self) -> None:
        load_errors = []
        for backend in self.backend_preference:
            try:
                if backend == "sentence_transformers" and self._load_with_sentence_transformers():
                    return
                if backend == "transformers" and self._load_with_transformers():
                    return
            except Exception as exc:
                load_errors.append(f"{backend}: {exc}")

        raise RuntimeError(
            "Failed to load the embedding model. Checked backends: "
            f"{', '.join(self.backend_preference)}. Errors: {' | '.join(load_errors)}"
        )

    def _mean_pool(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)

    def _encode_with_transformers(self, texts: Sequence[str], batch_size: int) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded_input = {key: value.to(self.device) for key, value in encoded_input.items()}

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            sentence_embeddings = self._mean_pool(model_output.last_hidden_state, encoded_input["attention_mask"])
            if self.normalize_embeddings:
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            all_embeddings.extend(sentence_embeddings.cpu().tolist())
        return all_embeddings

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        batch_size: int = 32,
        convert_to_list: bool = True,
    ) -> Union[List[float], List[List[float]]]:
        single_input = isinstance(texts, str)
        text_list = [texts] if single_input else list(texts)

        if not text_list:
            return []

        if self.backend == "sentence_transformers":
            embeddings = self.model.encode(
                text_list,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=convert_to_list,
            )
            if convert_to_list:
                embeddings = embeddings.tolist()
        else:
            embeddings = self._encode_with_transformers(text_list, batch_size=batch_size)

        return embeddings[0] if single_input else embeddings

    def batch_encode(self, texts: Sequence[str], batch_size: int = 32) -> List[List[float]]:
        return self.encode(texts, batch_size=batch_size, convert_to_list=True)

    def get_embedding_dimension(self) -> int:
        sample = self.encode(["dimension probe"])
        return len(sample[0])


if __name__ == "__main__":
    loader = EmbeddingModelLoader()
    sample_texts = [
        "Column: apt_id. Description: apartment identifier.",
        "Column: booking_date. Description: date of the booking.",
    ]
    vectors = loader.batch_encode(sample_texts)
    print(f"Loaded backend: {loader.backend}")
    print(f"Model: {loader.model_name}")
    print(f"Device: {loader.device}")
    print(f"Vector dimension: {len(vectors[0]) if vectors else 0}")
