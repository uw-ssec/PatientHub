# Evaluators

Evaluators in PatientHub assess the quality of therapy simulations, providing automated metrics and analysis of conversations between client and therapist agents.

## Available Evaluators

| Evaluator     | Key         | Description                                             |
| ------------- | ----------- | ------------------------------------------------------- |
| **LLM Judge** | `llm_judge` | Uses large language models to evaluate therapy sessions |

## Usage

### In Configuration

```yaml
evaluator:
  key: llm_judge
  config:
    model: gpt-4o
    criteria:
      - empathy
      - adherence
      - effectiveness
```

### In Code

```python
from patienthub.evaluators import EvaluatorRegistry

# List available evaluators
available = EvaluatorRegistry.list()

# Create an evaluator
evaluator = EvaluatorRegistry.create("llm_judge", config={
    "model": "gpt-4o"
})

# Evaluate a conversation
results = evaluator.evaluate(conversation_history)
```

## LLM Judge

The LLM Judge evaluator uses large language models to assess therapy sessions across multiple dimensions.

### Evaluation Criteria

The LLM Judge can evaluate sessions based on:

- **Empathy** - How well the therapist demonstrates understanding and compassion
- **Adherence** - Whether therapeutic techniques are properly applied
- **Effectiveness** - The potential therapeutic value of the session
- **Safety** - Appropriate handling of crisis situations
- **Rapport** - Quality of therapeutic alliance

### Configuration

```yaml
evaluator:
  key: llm_judge
  config:
    model: gpt-4o
    temperature: 0.3
    criteria:
      - empathy
      - adherence
      - effectiveness
```

### Output Format

```python
{
    "overall_score": 0.85,
    "criteria_scores": {
        "empathy": 0.9,
        "adherence": 0.8,
        "effectiveness": 0.85
    },
    "feedback": "The therapist demonstrated strong empathy...",
    "suggestions": ["Consider exploring...", "Could improve..."]
}
```

## Running Evaluations

### Command Line

```bash
python examples/evaluate.py \
    --session_path outputs/session.json \
    --evaluator llm_judge
```

### Batch Evaluation

```python
from patienthub.evaluators import EvaluatorRegistry

evaluator = EvaluatorRegistry.create("llm_judge")

sessions = load_sessions("outputs/")
results = []

for session in sessions:
    result = evaluator.evaluate(session)
    results.append(result)
```

## Creating Custom Evaluators

You can create custom evaluators by extending the base class:

```python
from patienthub.evaluators.base import Evaluator

class MyCustomEvaluator(Evaluator):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your evaluator

    def evaluate(self, conversation_history):
        # Perform evaluation
        return {
            "score": score,
            "feedback": feedback
        }
```

Then register it:

```python
from patienthub.evaluators import EvaluatorRegistry

EvaluatorRegistry.register("my_evaluator", MyCustomEvaluator)
```

## See Also

- [API Reference: Evaluators](../api/evaluators.md)
- [Creating Custom Evaluators](../contributing/new-evaluators.md)
- [Evaluation Guide](../guide/evaluation.md)
