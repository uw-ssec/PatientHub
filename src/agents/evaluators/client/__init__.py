from .conv import ConvEvaluator

EVALUATORS = {
    "conv": ConvEvaluator,
}


def get_client_evaluator(configs):
    eval_type = configs.eval_type
    print(f"Loading {eval_type} evaluator agent...")
    if eval_type in EVALUATORS:
        return EVALUATORS[eval_type](configs)
    else:
        raise ValueError(f"Unknown evaluator type: {eval_type}")


__all__ = [
    "ConvEvaluator",
]
