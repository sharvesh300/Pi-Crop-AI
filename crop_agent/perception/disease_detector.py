import numpy as np
import cv2
import platform

if platform.machine() in ["armv7l","aarch64"]:
    from tflite_runtime.interpreter import Interpreter
    BACKEND="tflite"
else:
    from tensorflow.lite import Interpreter
    BACKEND="tensorflow"



class TFLiteDiseaseDetector:
    def __init__(self, model_path, class_names):
        self.class_names = class_names
       
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.img_size = (224, 224)

    def preprocess(self, image):
        # The TFLite model includes mobilenet_v3.preprocess_input as the first
        # layer (injected during training), so it handles [0,255]→[-1,1]
        # scaling internally.  We must NOT manually normalise here.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        image = image.astype(np.float32)  # raw [0, 255] — model scales internally
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image):
        input_data = self.preprocess(image)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        probs = output[0]

        class_id = np.argmax(probs)

        return {
            "disease": self.class_names[class_id],
            "confidence": float(probs[class_id]),
        }
