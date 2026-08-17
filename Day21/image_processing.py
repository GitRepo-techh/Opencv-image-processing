import numpy as np
import cv2


# 1. Grayscale
def to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# 2. Blur
def apply_blur(img, method="gaussian", ksize=5, sigma=0):
    if ksize % 2 == 0:
        ksize += 1

    if method == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)
    elif method == "median":
        return cv2.medianBlur(img, ksize)
    elif method == "bilateral":
        return cv2.bilateralFilter(img, d=ksize, sigmaColor=75, sigmaSpace=75)
    else:
        raise ValueError(f"Unknown blur method: {method}")


# 3. Edge Detection
def detect_edges(img, threshold1=100, threshold2=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


# 4. Rotation
def rotate_image(img, angle=0, scale=1.0):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (w, h))


# 5. Brightness / Contrast Enhancement
def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# 6. Contour Detection (FIXED: Returns image and contour count)
def detect_contours(img, thresh_val=127, draw_all=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    mode = cv2.RETR_TREE if draw_all else cv2.RETR_EXTERNAL
    contours, _ = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output, len(contours)


# 7. Shape Detection (FIXED: Returns image and shape dictionary)
def detect_shapes(img, thresh_val=127):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()
    shape_counts = {
        "Triangle": 0,
        "Square": 0,
        "Rectangle": 0,
        "Pentagon": 0,
        "Circle": 0
    }

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        vertices = len(approx)

        if vertices == 3:
            shape = "Triangle"
            shape_counts["Triangle"] += 1
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h)
            if 0.95 <= ratio <= 1.05:
                shape = "Square"
                shape_counts["Square"] += 1
            else:
                shape = "Rectangle"
                shape_counts["Rectangle"] += 1
        elif vertices == 5:
            shape = "Pentagon"
            shape_counts["Pentagon"] += 1
        elif vertices > 5:
            shape = "Circle"
            shape_counts["Circle"] += 1
        else:
            shape = "Unknown"

        x, y, w, h = cv2.boundingRect(approx)
        cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)
        cv2.putText(output, shape, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return output, shape_counts


def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def scan_document(img, use_adaptive_threshold=True):
    orig = img.copy()
    ratio = img.shape[0] / 500.0
    resized = cv2.resize(img, (int(img.shape[1] / ratio), 500))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_contour = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            doc_contour = approx
            break

    if doc_contour is None:
        return orig

    pts = doc_contour.reshape(4, 2) * ratio
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    if use_adaptive_threshold:
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped = cv2.adaptiveThreshold(
            warped_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 10
        )
        warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    return warped


def sharpen(img):
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def flip_image(img, direction="horizontal"):
    flip_code = {"horizontal": 1, "vertical": 0, "both": -1}.get(direction, 1)
    return cv2.flip(img, flip_code)


def apply_threshold(img, thresh_val=127, method="binary"):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == "otsu":
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


def apply_sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                        [0.349, 0.686, 0.168],
                        [0.393, 0.769, 0.189]])
    sepia = cv2.transform(img, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)


def detect_color(image, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    return result, mask



def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    result = image.copy()
    for x, y, w, h in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            result, "Face", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    return result, len(faces)



def detect_faces_and_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    result = image.copy()
    total_eyes = 0

    for x, y, w, h in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = result[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        total_eyes += len(eyes)

        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    return result, len(faces), total_eyes