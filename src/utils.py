import os
import json
import yaml
from jinja2 import Template
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

load_dotenv(".env")


def load_json(path: str):
    return json.load(open(path, "r", encoding="utf-8"))


def load_yaml(path: str):
    try:
        with open(path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print("Error loading YAML file:", e)
        return {}


def load_prompts(role: str, agent_type: str, lang: str = "en"):
    path = f"data/prompts/{role}/{agent_type}.yaml"
    prompts = {}
    data = load_yaml(path)[lang]
    for k, v in data.items():
        if isinstance(v, str):
            prompts[k] = Template(v)
    return prompts


def parse_json_response(res):
    try:
        res = res.replace("```json", "").replace("```", "")
        res = json.loads(res)
        return res
    except Exception as e:
        print(f"Error while parsing JSON response: {e}")
        return res


def save_json(data, file_path: str):
    # Check if the directory exists, if not create it
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def append_json(item, file_path: str):
    existing = []
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    if isinstance(existing, list):
        existing.append(item)
    elif existing:
        existing = [existing, item]
    else:
        existing = [item]

    save_json(existing, file_path)


def get_model_client(configs):
    if configs["api_type"] == "local":
        hf_pipe = HuggingFacePipeline.from_model_id(
            model_id=configs["model_name"],
            task="text-generation",
            device=configs["device"],
            pipeline_kwargs={
                "max_new_tokens": configs["max_new_tokens"],
                "temperature": configs["temperature"],
                "repetition_penalty": configs["repetition_penalty"],
                "return_full_text": False,
            },
        )
        return init_chat_model(
            model_provider="huggingface", model=configs["model_name"], llm=hf_pipe
        )
    else:
        api_type = configs["api_type"]
        return init_chat_model(
            model=configs["model_name"],
            model_provider="openai",
            base_url=os.environ.get(f"{api_type}_BASE_URL"),
            api_key=os.environ.get(f"{api_type}_API_KEY"),
            temperature=configs["temperature"],
            max_tokens=configs["max_tokens"],
            max_retries=configs["max_retries"],
        )
