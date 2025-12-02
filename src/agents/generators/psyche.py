import json
from typing import Any, Dict

from omegaconf import DictConfig
from langchain_core.messages import SystemMessage

from src.utils import load_prompts, get_model_client, parse_json_response


class PsycheGenerator:
    def __init__(self, configs: DictConfig):
        self.configs = configs
        self.model_client = get_model_client(configs)
        self.prompts = load_prompts(
            role="generator", agent_type="psyche", lang=configs.lang
        )

    def generate_mfc_profile(
        self, diagnosis: str, age: int, sex: str
    ) -> Dict[str, Any]:
        prompt = self.prompts["MFC_Profile_generate_prompt"].render(
            diagnosis=diagnosis,
            age=age,
            sex=sex,
        )
        res = self.model_client.invoke([SystemMessage(content=prompt)])
        profile = parse_json_response(res.content)
        return profile

    def generate_mfc_history(
        self, diagnosis: str, age: int, sex: str, mfc_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        profile_json = json.dumps(mfc_profile, ensure_ascii=False, indent=2)
        prompt = self.prompts["MFC_History_generate_prompt"].render(
            diagnosis=diagnosis,
            age=age,
            sex=sex,
            mfc_profile_json=profile_json,
        )
        res = self.model_client.invoke([SystemMessage(content=prompt)])
        history = parse_json_response(res.content)
        return history

    def generate_mfc_behavior(
        self,
        diagnosis: str,
        age: int,
        sex: str,
        mfc_profile: Dict[str, Any],
        mfc_history: Dict[str, Any],
    ) -> Dict[str, Any]:
        profile_json = json.dumps(mfc_profile, ensure_ascii=False, indent=2)
        history_json = json.dumps(mfc_history, ensure_ascii=False, indent=2)
        prompt = self.prompts["MFC_Behavior_generate_prompt"].render(
            diagnosis=diagnosis,
            age=age,
            sex=sex,
            mfc_profile_json=profile_json,
            mfc_history_json=history_json,
        )
        res = self.model_client.invoke([SystemMessage(content=prompt)])
        behavior = parse_json_response(res.content)
        return behavior

    def generate_case(self, diagnosis: str, age: int, sex: str) -> Dict[str, Any]:
        mfc_profile = self.generate_mfc_profile(diagnosis=diagnosis, age=age, sex=sex)
        mfc_history = self.generate_mfc_history(
            diagnosis=diagnosis,
            age=age,
            sex=sex,
            mfc_profile=mfc_profile,
        )
        mfc_behavior = self.generate_mfc_behavior(
            diagnosis=diagnosis,
            age=age,
            sex=sex,
            mfc_profile=mfc_profile,
            mfc_history=mfc_history,
        )

        case = {
            "Diagnosis": diagnosis,
            "Age": age,
            "Sex": sex,
            "MFC-Profile": mfc_profile,
            "MFC-History": mfc_history.get("MFC-History", ""),
            "MFC-Behavior": mfc_behavior,
        }
        return case

