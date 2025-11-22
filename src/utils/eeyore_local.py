import os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class EeyoreLocalModel:
    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_dir = model_dir
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=True,
            local_files_only=True,
        )

        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            local_files_only=True,
        ).to(self.device)

        try:
            self.generation_config = GenerationConfig.from_pretrained(
                model_dir,
                local_files_only=True,
            )
        except Exception:
            self.generation_config = GenerationConfig()

        self.generation_config.max_new_tokens = max_new_tokens
        self.generation_config.temperature = temperature
        self.generation_config.top_p = top_p
        self.generation_config.do_sample = temperature > 0

    @classmethod
    def from_env_or_default(cls) -> "EeyoreLocalModel":
        model_dir = os.getenv("EEYORE_MODEL_DIR", "/data/public_checkpoints/Eeyore_llama3.1_8B")
        device = os.getenv("EEYORE_DEVICE")
        return cls(model_dir=model_dir, device=device)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: list of {'role': 'system'|'user'|'assistant', 'content': str}
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
            )

        generated_ids = output_ids[:, input_ids.shape[-1] :]
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        if text.lower().startswith("assistant"):
            parts = text.split("\n", 1)
            if len(parts) > 1:
                text = parts[1].lstrip()

        return text
