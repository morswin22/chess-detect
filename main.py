import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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

def find_squares(image):
    board_contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    marks = np.zeros_like(image)
    square_centers = list()

    for contour in board_contours:
        if 2000 < cv2.contourArea(contour) < 20000:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                pts = [pt[0].tolist() for pt in approx]

                # create same pattern for points, bottomright(1) , topright(2) , topleft(3) , bottomleft(4)
                index_sorted = sorted(pts, key=lambda x: x[0], reverse=True)

                if index_sorted[0][1]< index_sorted[1][1]:
                    cur=index_sorted[0]
                    index_sorted[0] =  index_sorted[1]
                    index_sorted[1] = cur

                if index_sorted[2][1]> index_sorted[3][1]:
                    cur=index_sorted[2]
                    index_sorted[2] =  index_sorted[3]
                    index_sorted[3] = cur

                # bottomright(1) , topright(2) , topleft(3) , bottomleft(4)
                pt1=index_sorted[0]
                pt2=index_sorted[1]
                pt3=index_sorted[2]
                pt4=index_sorted[3]

                x, y, w, h = cv2.boundingRect(contour)
                center_x=(x+(x+w))/2
                center_y=(y+(y+h))/2

                l1 = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                l2 = math.sqrt((pt2[0] - pt3[0])**2 + (pt2[1] - pt3[1])**2)
                l3 = math.sqrt((pt3[0] - pt4[0])**2 + (pt3[1] - pt4[1])**2)
                l4 = math.sqrt((pt1[0] - pt4[0])**2 + (pt1[1] - pt4[1])**2)

                lengths = [l1, l2, l3, l4]
                max_length = max(lengths)
                min_length = min(lengths)

                if (max_length - min_length) <= 35: # 20 for smaller boards, 50 for bigger, 35 works most of the time
                    square_centers.append([center_x,center_y,pt1,pt2,pt3,pt4])

                    cv2.line(marks, pt1, pt2, (255, 255, 0), 7)
                    cv2.line(marks, pt2, pt3, (255, 255, 0), 7)
                    cv2.line(marks, pt3, pt4, (255, 255, 0), 7)
                    cv2.line(marks, pt1, pt4, (255, 255, 0), 7)

    return square_centers, marks

def lerp(a, b, t):
    return a + (b - a) * t

stop = False
while not stop:
    ret, frame = capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bgr_image  = frame.copy()

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

    # Find squares
    square_centers, square_marks = find_squares(hough_dilation)

    square_marks_dilation = cv2.dilate(square_marks, np.ones((17, 17), np.uint8), iterations=1)

    # Look for larges contour (whole board)
    contours, _ = cv2.findContours(square_marks_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    biggest_area_image = np.zeros_like(square_marks_dilation)

    cv2.drawContours(biggest_area_image, largest_contour, -1, (255,255,255), 10)

    # Filter squares that are outside the biggest contour
    inside_squares=[square for square in square_centers if cv2.pointPolygonTest(largest_contour, (square[0], square[1]), measureDist=False) >= 0]

    # Sort squares
    sorted_coordinates = sorted(inside_squares, key=lambda x: x[1], reverse=True)
    current_group = [sorted_coordinates[0]]
    groups = list()

    for coord in sorted_coordinates[1:]:
        if abs(coord[1] - current_group[-1][1]) < 50: # y_error_threshold
            current_group.append(coord)
        else:
            groups.append(current_group)
            current_group = [coord]
    groups.append(current_group)

    for group in groups:
        group.sort(key=lambda x: x[0])

    sorted_coordinates = [coord for group in groups for coord in group]

    dxs = list()
    for i in range(len(sorted_coordinates) - 1):
        x1, y1, *_ = sorted_coordinates[i]
        x2, y2, *_ = sorted_coordinates[i+1]
        dx = x2 - x1
        if dx < 0:
            continue
        dxs.append(dx)

    x_gap = np.median(dxs)

    # Fill missing squares
    for i in range(len(sorted_coordinates) - 1, 1, -1):
        x1, y1, *_ = sorted_coordinates[i-1]
        x2, y2, *_ = sorted_coordinates[i]
        dx = x2 - x1
        if dx < x_gap*1.5:
            continue
        squares_missing = dx // x_gap
        for j in range(int(squares_missing)):
            t = (j+1) / (squares_missing+1)
            sorted_coordinates.insert(i, (lerp(x1, x2, t), lerp(y1, y2, t)))

    # Display squares
    for index, coord in enumerate(sorted_coordinates):
        x, y = coord[0], coord[1]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 0, 255)
        thickness = 2

        cv2.putText(bgr_image, str(index), np.array((x, y), np.uint), font, font_scale, color, thickness)

    cv2.imshow("Live Stream", bgr_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop = True
        # cv2.imwrite("frame.png", frame)
        break

capture.release()
cv2.destroyAllWindows()

# %%

