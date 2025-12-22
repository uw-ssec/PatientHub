import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

load_dotenv(".env")


def get_chat_model(configs):
    # Support both DictConfig (attribute access) and dict (key access)
    def get(name, default=None):
        if isinstance(configs, dict):
            return configs.get(name, default)
        return getattr(configs, name, default)

    model_type = get("model_type")
    model_name = get("model_name")

    if model_type == "local":
        hf_pipe = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            device=get("device"),
            pipeline_kwargs={
                "max_new_tokens": get("max_new_tokens"),
                "temperature": get("temperature"),
                "repetition_penalty": get("repetition_penalty"),
                "return_full_text": False,
            },
        )
        return init_chat_model(
            model_provider="huggingface", model=model_name, llm=hf_pipe
        )
    if model_type == "huggingface":
        return ChatHuggingFace.from_model_id(
            model_id=model_name,
            task="text-generation",
            device=get("device"),
            model_kwargs={
                "max_new_tokens": get("max_new_tokens"),
                "temperature": get("temperature"),
                "repetition_penalty": get("repetition_penalty"),
                "return_full_text": False,
            },
        )
    if model_type == "LAB":
        return init_chat_model(
            model=model_name,
            model_provider="openai",
            base_url=os.environ.get(f"{model_type}_BASE_URL"),
            api_key=os.environ.get(f"{model_type}_API_KEY"),
            temperature=get("temperature"),
            max_tokens=get("max_tokens"),
            max_retries=get("max_retries"),
        )


def get_reranker_model(configs):
    # Support both DictConfig (attribute access) and dict (key access)
    def cfg(name, default=None):
        if isinstance(configs, dict):
            return configs.get(name, default)
        return getattr(configs, name, default)

    model_type = cfg("model_type")
    model_name = cfg("model_name")
    if model_type not in ("huggingface", "local") or not model_name:
        return None, None, None

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception:
        return None, None, None

    device_index = cfg("device", 0)
    try:
        device_index = int(device_index)
    except (TypeError, ValueError):
        device_index = 0

    if torch.cuda.is_available() and device_index >= 0:
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception:
        return None, None, None


def rerank_query_passages(
    tokenizer, model, device, query: str, passages: list[str], max_length: int = 512
):
    """Score (query, passage) pairs with a cross-encoder model.

    Returns a list of scores (higher means more relevant), or None on failure.
    """
    if tokenizer is None or model is None or not passages:
        return None

    try:
        import torch
    except Exception:
        return None

    pairs = list(zip([query] * len(passages), passages))
    try:
        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, return_dict=True)
            logits = outputs.logits.view(-1).float()
            scores = torch.sigmoid(logits).tolist()
        return scores
    except Exception:
        return None
