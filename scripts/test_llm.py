from crop_agent.memory.vector_store import VectorStore
from crop_agent.memory.metadata_store import MetadataStore
from crop_agent.llm.llm_client import LLMEngine
from crop_agent.utils import SYSTEM_CONFIG

vector_store = VectorStore(SYSTEM_CONFIG)
metadata_store = MetadataStore()
llm = LLMEngine()

mock_case = {
    "crop": "Tomato",
    "disease": "Leaf Blight",
    "severity": "Medium",
    "confidence": 0.88,
    "temperature": 32,
    "humidity": 78,
}

query_text = (
    f"{mock_case['crop']} {mock_case['disease']} "
    f"{mock_case['severity']} "
    f"{mock_case['temperature']}C "
    f"{mock_case['humidity']}%"
)

indices, scores = vector_store.search(query_text)

similar_cases = [metadata_store.get_case(idx + 1) for idx in indices]

decision = llm.generate_decision(mock_case, similar_cases)

print("Decision:", decision)
