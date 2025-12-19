from .activeListening import ActiveListeningEvaluator
from .cbt import CBTEvaluator

EVALUATORS = {
    "activeListening": ActiveListeningEvaluator,
    "cbt": CBTEvaluator,
}


def get_therapist_evaluator(configs):
    eval_type = configs.eval_type
    print(f"Loading therapist evaluator agent for {eval_type}...")
    if eval_type in EVALUATORS:
        return EVALUATORS[eval_type](configs)
    else:
        raise ValueError(f"Unknown therapist evaluator type: {eval_type}")
