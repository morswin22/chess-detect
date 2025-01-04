import cv2
import numpy as np
import matplotlib.pyplot as plt

class Livestream:
    def __init__(self, url):
        self.capture = cv2.VideoCapture(url)

    def release(self):
        self.capture.release()

    def read(self):
        return self.capture.read()

class Image:
    def __init__(self, path):
        self.image = cv2.imread(path)

    def release(self):
        pass

    def read(self):
        return True, self.image

# capture = Livestream("https://192.168.1.34:8080/video")
capture = Image("frame.png")
# capture = Image("test-10.jpeg")

stop = False
while not stop:
    ret, frame = capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    # frame = cv2.resize(frame, (1500, 2000))

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Blur
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # OTSU tresholding
    ret, otsu_binary = cv2.threshold(blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Closing
    closed_image = cv2.morphologyEx(otsu_binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    # Canny edge detection
    canny = cv2.Canny(closed_image, 20, 255)

    # Dilation
    img_dilation = cv2.dilate(canny, np.ones((7,7), np.uint8), iterations=1)

    # Hough Lines
    lines = cv2.HoughLinesP(img_dilation, 1, np.pi / 180, threshold=200, minLineLength=150, maxLineGap=100)
    hough_image = np.zeros_like(img_dilation)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 90
            if angle < 10 or angle > 80:
                cv2.line(hough_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    hough_dilation = cv2.dilate(hough_image, np.ones((3, 3), np.uint8), iterations=1)

    cv2.imshow("Live Stream", hough_dilation)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop = True
        # cv2.imwrite("frame.png", frame)
        break

capture.release()
cv2.destroyAllWindows()

# %%

