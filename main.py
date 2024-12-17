import cv2

url = "https://192.168.1.34:8080/video"

capture = cv2.VideoCapture(url)

while True:
    ret, frame = capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Live Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

# %%

