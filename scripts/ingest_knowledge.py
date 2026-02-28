import json
from crop_agent.memory.vector_store import VectorStore
from crop_agent.memory.metadata_store import MetadataStore

vector_store = VectorStore("config/system_config.yaml")
metadata_store = MetadataStore()

with open("data/seed/sample.json") as f:
    cases = json.load(f)

for case in cases:
    text = (
        f"{case['crop']} {case['disease']} "
        f"{case['severity']} "
        f"{case['temperature']}C "
        f"{case['humidity']}% "
        f"{case['treatment_applied']} "
        f"{case['outcome']}"
    )

    vector_store.add(text)
    metadata_store.add_case(text)

vector_store.save()

print("Memory seeded successfully.")
