from .active_listening import ActiveListeningFeedback
from .cbt import CBTFeedback
from .character import CharacterFeedback


def get_feedback_schema(eval_type: str):
    if eval_type == "active_listening":
        return ActiveListeningFeedback
    elif eval_type == "cbt":
        return CBTFeedback
    elif eval_type == "character":
        return CharacterFeedback
    else:
        raise ValueError(f"Unknown evaluator type: {eval_type}")


__all__ = [
    "ActiveListeningFeedback",
    "CBTFeedback",
    "CharacterFeedback",
]
