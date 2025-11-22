import os
import json
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


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
            dtype=dtype,
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

'''
    因为eeyore模型训练过程中就使用的固定格式
    所以改成它习惯使用的 HF chat_template 格式
'''
class EeyoreChatModel(BaseChatModel):
    local_model: EeyoreLocalModel

    def __init__(
        self,
        local_model: Optional[EeyoreLocalModel] = None,
        **kwargs: Any,
    ):
        if local_model is None:
            local_model = EeyoreLocalModel.from_env_or_default()
        super().__init__(local_model=local_model, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "eeyore-local"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Dict[str, Any],
    ) -> ChatResult:
        chat_messages: List[Dict[str, str]] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            chat_messages.append({"role": role, "content": m.content})

        text = self.local_model.generate(chat_messages)
        msg = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # 由于各种模型都使用self.model_client.with_structured_output(response_format)
    # 因此采用妥协的方法来进行输出，可能实现得比较不优雅……
    def with_structured_output(
        self,
        schema: Any,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ):
        def parse(text: str):
            cleaned = text.replace("```json", "").replace("```", "").strip()

            if isinstance(schema, type) and issubclass(schema, BaseModel):
                try:
                    return schema.model_validate_json(cleaned)
                except Exception:
                    pass
                try:
                    data = json.loads(cleaned)
                    return schema.model_validate(data)
                except Exception:
                    pass

                fields = getattr(schema, "model_fields", None) or getattr(
                    schema, "__fields__", {}
                )
                if "content" in fields:
                    try:
                        return schema(content=text)
                    except Exception:
                        pass
                try:
                    return schema()
                except Exception:
                    return schema

            try:
                return json.loads(cleaned)
            except Exception:
                return {"content": text}

        class Wrapper:
            def __init__(self, base: "EeyoreChatModel"):
                self.base = base

            def invoke(self, messages: List[BaseMessage]):
                msg = self.base.invoke(messages)
                parsed = parse(msg.content)
                if include_raw:
                    return {"raw": msg, "parsed": parsed, "parsing_error": None}
                return parsed

        return Wrapper(self)


def get_eeyore_chat_model() -> EeyoreChatModel:
    local = EeyoreLocalModel.from_env_or_default()
    return EeyoreChatModel(local)
