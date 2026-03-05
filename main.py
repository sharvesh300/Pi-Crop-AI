"""
pi-crop-ai — main pipeline entry point.

Pipeline:
  1. Load image
  2. TFLiteDiseaseDetector  → disease label + confidence
  3. SeverityEstimator      → severity % + class
  4. DecisionAgent.decide() → decision + reason + treatment plan

Usage:
  python main.py <image_path> [crop_name]

  image_path  Path to the leaf image (required)
  crop_name   Optional crop name passed to the agent (default: "Unknown")
"""

import sys
from pathlib import Path

import cv2

from crop_agent.perception.disease_detector import TFLiteDiseaseDetector
from crop_agent.perception.severity_estimator import SeverityEstimator
from crop_agent.agent.crop_agent import DecisionAgent
from crop_agent.llm.prompt_loader import _get_disease_block
from crop_agent.utils import load_yaml, MODEL_CONFIG

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

_model_cfg = load_yaml(MODEL_CONFIG)

MODEL_PATH = Path(_model_cfg["cnn"]["model_path"])
CONF_THRESH = float(_model_cfg["cnn"].get("confidence_threshold", 0.75))

# Disease class names — must match the alphabetical subdirectory order used by
# image_dataset_from_directory during training (confirmed by training notebook).
# The TFLite model embeds mobilenet_v3.preprocess_input, so inference must feed
# raw float32 [0, 255] pixels — see TFLiteDiseaseDetector.preprocess().
#   0: bacterial_blight
#   1: cercospora_leaf_blight
#   2: downey_mildew
#   3: frogeye
#   4: healthy
#   5: potassium_deficiency
#   6: soybean_rust
#   7: target_spot
CLASS_NAMES = [
    "Bacterial Blight",
    "Cercospora Leaf Blight",
    "Downy Mildew",
    "Frogeye Leaf Spot",
    "Healthy",
    "Potassium Deficiency",
    "Soybean Rust",
    "Target Spot",
]

# Map severity estimator class → agent severity level
_SEVERITY_MAP = {
    "mild": "low",
    "moderate": "medium",
    "severe": "high",
    "critical": "high",
}


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def run_pipeline(image_path: str, crop: str = "Unknown") -> dict:
    """
    Run the full perception + decision pipeline on a single image.

    Returns the merged result dict from DecisionAgent.decide().
    """

    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    print(f"\n[1/4] Image loaded : {image_path}")

    # 2. Disease detection
    detector = TFLiteDiseaseDetector(
        model_path=str(MODEL_PATH),
        class_names=CLASS_NAMES,
    )
    detection = detector.predict(img)
    disease = detection["disease"]
    confidence = detection["confidence"]

    print(f"[2/4] Disease      : {disease}  (confidence: {confidence:.2%})")
    print()
    print(_get_disease_block(disease))
    print()

    if confidence < CONF_THRESH:
        print(
            f"      ⚠  Confidence below threshold ({CONF_THRESH:.0%}). "
            "Proceeding with low-confidence detection."
        )

    # 3. Severity estimation
    estimator = SeverityEstimator()
    severity_result = estimator.estimate(image_path, visualize=False)
    severity_percent = severity_result["severity_percent"]
    severity_class = severity_result["severity_class"]
    agent_severity = _SEVERITY_MAP[severity_class]

    print(
        f"[3/4] Severity     : {severity_percent:.1f}%  ({severity_class} → agent level: {agent_severity})"
    )

    # 4. Agent decision
    case_context = {
        "crop": crop,
        "disease": disease,
        "severity": agent_severity,
        "confidence": confidence,
    }

    print("[4/4] Running DecisionAgent ...")
    agent = DecisionAgent()
    result = agent.decide(case_context)

    # attach perception fields so print_result can display them
    result["_crop"] = crop
    result["_disease"] = disease
    result["_confidence"] = confidence
    result["_severity_percent"] = severity_percent
    result["_severity_class"] = severity_class

    return result


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def print_result(result: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  CROP     : {result.get('_crop', 'N/A')}")
    print(
        f"  DISEASE  : {result.get('_disease', 'N/A')}  "
        f"(confidence: {result.get('_confidence', 0):.2%})"
    )
    print(
        f"  SEVERITY : {result.get('_severity_percent', 0):.1f}%  "
        f"({result.get('_severity_class', 'N/A')})"
    )
    print("-" * 60)
    print("  DECISION :", result.get("decision"))
    print("  REASON   :", result.get("reason"))
    print("  SAFE     :", result.get("safe"))
    if result.get("override"):
        print("  OVERRIDE : yes — safety validator replaced LLM decision")
    if result.get("requires_confirmation"):
        print("  ACTION   : requires human confirmation before applying")

    plan = result.get("plan", [])
    if plan:
        print("\n  Treatment plan:")
        for step in plan:
            print(f"    {step['step']}. [{step['action']}] {step['details']}")
    else:
        print("\n  No treatment plan generated.")
    print("=" * 60 + "\n")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [crop_name]")
        sys.exit(1)

    image_path = sys.argv[1]
    crop_name = sys.argv[2] if len(sys.argv) >= 3 else "Unknown"

    result = run_pipeline(image_path, crop=crop_name)
    print_result(result)
