"""
pi-crop-ai — main pipeline entry point (Webcam + Image mode)

Pipeline
--------
1. Capture image from webcam OR load image from path
2. Disease detection (TFLite MobileNetV3)
3. Severity estimation
4. Decision agent reasoning
"""

import sys
from pathlib import Path
import cv2
import time

from crop_agent.perception.disease_detector import TFLiteDiseaseDetector
from crop_agent.perception.severity_estimator import SeverityEstimator
from crop_agent.agent.crop_agent import DecisionAgent
from crop_agent.llm.prompt_loader import _get_disease_block
from crop_agent.utils import load_yaml, MODEL_CONFIG


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_model_cfg = load_yaml(MODEL_CONFIG)

MODEL_PATH = Path(_model_cfg["cnn"]["model_path"])
CONF_THRESH = float(_model_cfg["cnn"].get("confidence_threshold", 0.75))


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

_SEVERITY_MAP = {
    "mild": "low",
    "moderate": "medium",
    "severe": "high",
    "critical": "high",
}


# ---------------------------------------------------------------------------
# Webcam Capture
# ---------------------------------------------------------------------------

def capture_from_webcam(save_path="captured_leaf.jpg"):
    """
    Capture image from webcam and return saved image path
    """

    print("Searching for camera...")

    for index in range(5):

        cap = cv2.VideoCapture(index)

        if cap.isOpened():

            print(f"Camera detected at /dev/video{index}")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            # allow camera auto exposure to adjust
            time.sleep(2)

            ret, frame = cap.read()

            if not ret:
                cap.release()
                raise RuntimeError("Failed to capture frame from webcam")

            cv2.imwrite(save_path, frame)

            print(f"Image saved → {save_path}")

            cap.release()

            return save_path

        cap.release()

    raise RuntimeError("No camera detected. Check USB camera connection.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(image_path: str, crop: str = "Unknown") -> dict:

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    print(f"\n[1/4] Image loaded : {image_path}")

    # Disease Detection
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
            f"⚠ Confidence below threshold ({CONF_THRESH:.0%}). "
            "Proceeding with low-confidence detection."
        )

    # Severity Estimation
    estimator = SeverityEstimator()

    severity_result = estimator.estimate(image_path, visualize=False)

    severity_percent = severity_result["severity_percent"]
    severity_class = severity_result["severity_class"]

    agent_severity = _SEVERITY_MAP[severity_class]

    print(
        f"[3/4] Severity     : {severity_percent:.1f}%  "
        f"({severity_class} → agent level: {agent_severity})"
    )

    # Decision Agent
    case_context = {
        "crop": crop,
        "disease": disease,
        "severity": agent_severity,
        "confidence": confidence,
    }

    print("[4/4] Running DecisionAgent ...")

    agent = DecisionAgent()

    result = agent.decide(case_context)

    result["_crop"] = crop
    result["_disease"] = disease
    result["_confidence"] = confidence
    result["_severity_percent"] = severity_percent
    result["_severity_class"] = severity_class

    return result


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------

def print_result(result: dict):

    print("\n" + "=" * 60)

    print(f"  CROP     : {result.get('_crop')}")
    print(
        f"  DISEASE  : {result.get('_disease')} "
        f"(confidence: {result.get('_confidence'):.2%})"
    )
    print(
        f"  SEVERITY : {result.get('_severity_percent'):.1f}% "
        f"({result.get('_severity_class')})"
    )

    print("-" * 60)

    print("  DECISION :", result.get("decision"))
    print("  REASON   :", result.get("reason"))
    print("  SAFE     :", result.get("safe"))

    if result.get("override"):
        print("  OVERRIDE : yes — safety validator replaced LLM decision")

    if result.get("requires_confirmation"):
        print("  ACTION   : requires human confirmation")

    plan = result.get("plan", [])

    if plan:
        print("\n  Treatment plan:")
        for step in plan:
            print(f"    {step['step']}. [{step['action']}] {step['details']}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # If image path provided → use file
    if len(sys.argv) >= 2 and Path(sys.argv[1]).exists():

        image_path = sys.argv[1]
        crop_name = sys.argv[2] if len(sys.argv) >= 3 else "Unknown"

    else:
        print("\n📷 No image path provided → capturing from webcam")

        image_path = capture_from_webcam("/home/saro/Desktop/Pi-Crop-AI/data/images/captured_leaf.jpg")

        crop_name = "Unknown"

    result = run_pipeline(image_path, crop=crop_name)

    print_result(result)