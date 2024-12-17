import cv2

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

while True:
    ret, frame = capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Live Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("frame.png", frame)
        break

capture.release()
cv2.destroyAllWindows()

# %%

