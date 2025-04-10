import cv2
import numpy as np

# Keypoints (name, x, y, score)
keypoints = [
    ("Nose", 641.81, 462.67, 0.93),
    ("L_Eye", 667.13, 422.06, 0.95),
    ("R_Eye", 623.36, 430.15, 0.97),
    ("L_Ear", 782.41, 385.24, 0.99),
    ("R_Ear", 642.39, 414.67, 0.70),
    ("L_Shoulder", 924.42, 528.05, 0.92),
    ("R_Shoulder", 665.23, 538.68, 0.84),
    ("L_Elbow", 1197.34, 584.01, 0.94),
    ("R_Elbow", 507.11, 465.09, 0.88),
    ("L_Wrist", 356.90, 361.53, 1.03),
    ("R_Wrist", 1452.61, 655.33, 0.96),
    ("L_Hip", 903.56, 1023.63, 0.83),
    ("R_Hip", 974.09, 1065.43, 0.64),
    ("L_Knee", 579.39, 1391.28, 0.97),
    ("R_Knee", 780.33, 1488.72, 1.01),
    ("L_Ankle", 691.16, 1864.86, 0.91),
    ("R_Ankle", 1187.69, 1565.69, 0.87),
]

# Only keep keypoints we want (no face except Nose)
keep = {"Nose", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
        "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"}

keypoints = [kp for kp in keypoints if kp[0] in keep]

# Fix swapped hips based on their positions
def fix_flipped_hips(kps):
    kp_dict = {kp[0]: kp for kp in kps}
    if kp_dict["L_Hip"][1] > kp_dict["R_Hip"][1]:  # x of L_Hip > x of R_Hip ⇒ flipped
        # Swap values
        l = kp_dict["L_Hip"]
        r = kp_dict["R_Hip"]
        kp_dict["L_Hip"] = ("L_Hip", r[1], r[2], r[3])
        kp_dict["R_Hip"] = ("R_Hip", l[1], l[2], l[3])
    return [kp_dict[k] for k in kp_dict]

keypoints = fix_flipped_hips(keypoints)

# Skeleton pairs using remaining keypoints
skeleton = [
    (0, 1), (0, 2),
    (1, 3), (3, 5),
    (2, 4), (4, 6),
    (1, 7), (2, 8),
    (7, 8), (7, 9), (9, 11),
    (8, 10), (10, 12)
]

# Load image
image_path = "C:/Users/helfa/Pictures/image.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image from: {image_path}")

overlay = image.copy()
output = image.copy()
threshold = 0.5

# Blend colors based on connection order
def blend_color(i, total):
    t = i / total
    return (
        int(255 * (1 - t)),  # Blue → Red
        int(64 + 128 * t),   # Dim → Brighter Green
        int(255 * t)
    )

# Draw skeleton
for idx, (i, j) in enumerate(skeleton):
    if i >= len(keypoints) or j >= len(keypoints):
        continue
    x1, y1, s1 = keypoints[i][1], keypoints[i][2], keypoints[i][3]
    x2, y2, s2 = keypoints[j][1], keypoints[j][2], keypoints[j][3]
    if s1 > threshold and s2 > threshold:
        color = blend_color(idx, len(skeleton))
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=8, lineType=cv2.LINE_AA)

# Draw glowing joints
for name, x, y, score in keypoints:
    if score > threshold:
        center = (int(x), int(y))
        cv2.circle(overlay, center, 16, (255, 255, 255), -1, cv2.LINE_AA)  # Outer glow
        cv2.circle(overlay, center, 10, (0, 0, 0), -1, cv2.LINE_AA)        # Shadow
        cv2.circle(overlay, center, 6, (0, 255, 255), -1, cv2.LINE_AA)     # Core

# Blend overlay
cv2.addWeighted(overlay, 0.8, output, 0.2, 0, output)

# Display in resizable window
cv2.namedWindow("Pose Visualization", cv2.WINDOW_NORMAL)
cv2.imshow("Pose Visualization", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
