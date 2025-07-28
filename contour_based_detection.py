import cv2
import numpy as np

# === Load your RGB and depth images ===
rgb = cv2.imread('demo_data/T/color_1920x1080.png')                  # RGB image
depth = cv2.imread('demo_data/T/depth_640x576.png', cv2.IMREAD_UNCHANGED)  # Depth in mm or meters

img = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)  # Optional: resize for speed

# Convert to HSV and isolate bright/white regions
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# White threshold in HSV
lower_white = np.array([0, 0, 180])
upper_white = np.array([180, 50, 255])
mask = cv2.inRange(hsv, lower_white, upper_white)

# Clean up with morphology
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 1000:
        continue

    # Optional: use Hu moments to filter T shape
    moments = cv2.HuMoments(cv2.moments(cnt)).flatten()

    # Bounding box and angle
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Draw results
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    cv2.putText(img, "T candidate", tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

cv2.imshow("T detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()