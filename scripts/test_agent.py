"""
End-to-end smoke test for the DecisionAgent pipeline.

Runs three mock cases covering low / medium / high severity to exercise
memory retrieval, LLM decision, safety validation, and treatment planning.

Usage:
    uv run scripts/test_agent.py
"""

import json
from crop_agent.agent.crop_agent import DecisionAgent

agent = DecisionAgent()

MOCK_CASES = [
    {
        "label": "Low severity — medicine should be blocked",
        "case": {
            "crop": "Tomato",
            "disease": "Powdery Mildew",
            "severity": "low",
            "confidence": 0.72,
            "temperature": 28,
            "humidity": 55,
        },
    },
    {
        "label": "Medium severity — full pipeline",
        "case": {
            "crop": "Tomato",
            "disease": "Leaf Blight",
            "severity": "medium",
            "confidence": 0.88,
            "temperature": 32,
            "humidity": 78,
        },
    },
    {
        "label": "High severity — monitoring should be blocked",
        "case": {
            "crop": "Pepper",
            "disease": "Bacterial Spot",
            "severity": "high",
            "confidence": 0.95,
            "temperature": 35,
            "humidity": 85,
        },
    },
]


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


for mock in MOCK_CASES:
    print_section(mock["label"])
    print("Input:", json.dumps(mock["case"], indent=2))

    result = agent.decide(mock["case"])

    print("\nDecision  :", result.get("decision"))
    print("Reason    :", result.get("reason"))

    plan = result.get("plan", [])
    if plan:
        print("\nPlan:")
        for step in plan:
            print(
                f"  Step {step.get('step')}: [{step.get('action')}] {step.get('details', '')}"
            )
    else:
        print("\nPlan      : (none — LLM did not produce a parseable plan)")
