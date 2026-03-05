from rembg import remove
import cv2
import numpy as np


class SeverityEstimator:
    def remove_background(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        output = remove(rgb)

        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGRA)

        alpha = output[:, :, 3]

        mask = np.where(alpha > 0, 255, 0).astype(np.uint8)

        kernel = np.ones((7, 7), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        leaf = cv2.bitwise_and(img, img, mask=mask)

        return leaf, mask

    def detect_disease(self, leaf, leaf_mask):
        # convert to float for index calculation
        leaf_float = leaf.astype("float32")

        B, G, R = cv2.split(leaf_float)

        # Excess Green index
        exg = 2 * G - R - B

        # normalize for visualization
        exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        # diseased pixels usually have LOW ExG
        _, disease_mask = cv2.threshold(exg_norm, 80, 255, cv2.THRESH_BINARY_INV)

        disease_mask = cv2.bitwise_and(disease_mask, leaf_mask)

        # remove noise
        kernel_small = np.ones((5, 5), np.uint8)
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel_small)

        # stronger lesion clustering: merge nearby pixels into real lesion regions
        kernel_large = np.ones((9, 9), np.uint8)
        disease_mask = cv2.dilate(disease_mask, kernel_large, iterations=2)
        disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel_large)

        # contour filtering: keep only clusters large enough to be real lesions
        clean_mask = np.zeros_like(disease_mask)

        contours, _ = cv2.findContours(
            disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 250:  # remove noise
                continue

            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

        disease_mask = clean_mask

        return disease_mask

    def compute_severity(self, leaf_mask, disease_mask):
        total_leaf_pixels = cv2.countNonZero(leaf_mask)

        total_disease_pixels = cv2.countNonZero(disease_mask)

        if total_leaf_pixels == 0:
            return 0.0

        severity_percent = (total_disease_pixels / total_leaf_pixels) * 100

        return severity_percent

    def classify_severity(self, severity_percent):
        if severity_percent < 5:
            return "mild"

        elif severity_percent < 20:
            return "moderate"

        elif severity_percent < 50:
            return "severe"

        else:
            return "critical"

    def highlight_disease(self, leaf, disease_mask):
        output = leaf.copy()

        contours, _ = cv2.findContours(
            disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < 400:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return output

    def estimate(self, image_path, visualize=True):
        img = cv2.imread(image_path)

        leaf, leaf_mask = self.remove_background(img)

        disease_mask = self.detect_disease(leaf, leaf_mask)

        severity_percent = self.compute_severity(leaf_mask, disease_mask)

        severity_class = self.classify_severity(severity_percent)

        highlighted = self.highlight_disease(leaf, disease_mask)

        if visualize:
            canvas = np.hstack(
                [
                    cv2.resize(img, (400, 400)),
                    cv2.resize(leaf, (400, 400)),
                    cv2.resize(
                        cv2.cvtColor(disease_mask, cv2.COLOR_GRAY2BGR), (400, 400)
                    ),
                    cv2.resize(highlighted, (400, 400)),
                ]
            )

            cv2.imshow("Severity Analysis", canvas)

            print("Severity:", round(severity_percent, 2), "%")
            print("Class:", severity_class)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {"severity_percent": severity_percent, "severity_class": severity_class}
