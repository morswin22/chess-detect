import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
import chess

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

def image_to_board(idx):
    rank = 7 - idx % 8
    file = idx // 8
    return rank * 8 + file

def board_to_image(i):
    rank = 7 - i // 8
    file = i % 8
    return file * 8 + rank

board = chess.Board()
board_state = dict()
for i in chess.SQUARES:
    piece = board.piece_at(i)
    if piece is not None:
        board_state[board_to_image(i)] = int(piece.color)

last_frame = None

stop = False
while not stop:
    ret, frame = capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    if last_frame is None:
        last_frame = np.zeros_like(frame)

    diff = cv2.absdiff(last_frame, frame)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    total_pixels = thresh.size
    differing_pixels = np.count_nonzero(thresh)
    ratio = differing_pixels / total_pixels

    last_frame = frame
    if ratio < 1e-4:
        continue

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bgr_image  = frame.copy()

    # Blur
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Binary tresholding
    binary_image = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Closing
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # Canny edge detection
    canny = cv2.Canny(closed_image, 20, 255)

    # Dilation
    img_dilation = cv2.dilate(canny, np.ones((7,7), np.uint8), iterations=1)

    # Hough Lines
    lines = cv2.HoughLinesP(img_dilation, 1, np.pi / 180, threshold=800, minLineLength=500, maxLineGap=25)
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

    if len(contours) == 0:
        continue

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
    if math.isnan(x_gap) or int(x_gap) == 0:
        continue

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

        cv2.putText(bgr_image, str(image_to_board(index)), np.array((x, y), np.uint), font, font_scale, color, thickness)

    if len(sorted_coordinates) != len(chess.SQUARES):
        cv2.imshow("Live Stream", bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop = True
            break
        continue

    # Find empty and occupied squares
    average_colors = dict()
    for idx, square in enumerate(sorted_coordinates):
        cx, cy, *_ = square
        y_indices, x_indices = np.ogrid[:img_dilation.shape[0], :img_dilation.shape[1]]
        distance_squared = (x_indices - cx) ** 2 + (y_indices - cy) ** 2
        count = np.sum((distance_squared <= (x_gap*0.4) ** 2) & (img_dilation == 255))
        if count > 1500:
            colors_in_circle = bgr_image[distance_squared <= (x_gap*0.4) ** 2]
            if len(colors_in_circle) > 0:
                avg_color = np.mean(colors_in_circle, axis=0)
                average_colors[idx] = avg_color

    # Cluster colors into "white" and "black" pieces
    valid_colors = np.array(list(average_colors.values()))
    new_board_state = dict()

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(valid_colors)
    # (0.299*R + 0.587*G + 0.114*B)
    luminance = [0.299*r + 0.587*g + 0.114*b for b, g, r in kmeans.cluster_centers_]
    clustered = kmeans.predict(valid_colors)
    for i, idx in enumerate(average_colors.keys()):
        new_board_state[idx] = int(clustered[i] if luminance[1] > luminance[0] else 1 - clustered[i])

    if board_state != new_board_state:
        added_squares = set(new_board_state.keys()).difference(set(board_state.keys()))
        removed_squares = set(board_state.keys()).difference(set(new_board_state.keys()))
        intersection_keys = set(board_state.keys()).intersection(set(new_board_state.keys()))
        changed_squares = {idx for idx in intersection_keys if board_state[idx] != new_board_state[idx]}
        move = None
        if len(removed_squares) == len(added_squares) == 1:
            # Move
            from_, to = map(image_to_board, (removed_squares.pop(), added_squares.pop()))
            move = chess.Move.from_uci(chess.SQUARE_NAMES[from_] + chess.SQUARE_NAMES[to])
        elif len(removed_squares) == 2 and len(added_squares) == 1:
            # this is possibly En passant
            pass # TODO: exd6e.p
        elif len(removed_squares) == 1 and len(changed_squares) == 1:
            # Capture
            from_, to = map(image_to_board, (removed_squares.pop(), changed_squares.pop()))
            move = chess.Move.from_uci(chess.SQUARE_NAMES[from_] + chess.SQUARE_NAMES[to])
        elif len(removed_squares) == len(added_squares) == 2:
            # this is possibly Castling
            king_square = None
            rook_square = None
            for idx in removed_squares:
                piece = board.piece_at(image_to_board(idx))
                if piece is None:
                    break
                if piece.piece_type == chess.KING:
                    king_square = idx
                elif piece.piece_type == chess.ROOK:
                    rook_square = idx

            if king_square is not None and rook_square is not None:
                to_squares = map(lambda idx: chess.SQUARE_NAMES[image_to_board(idx)], added_squares)
                from_ = chess.SQUARE_NAMES[image_to_board(king_square)]
                if "c1" in to_squares and "d1" in to_squares:
                    move = chess.Move.from_uci(from_ + "c1")
                elif "g1" in to_squares and "f1" in to_squares:
                    move = chess.Move.from_uci(from_ + "g1")
                elif "c8" in to_squares and "d8" in to_squares:
                    move = chess.Move.from_uci(from_ + "c8")
                elif "g8" in to_squares and "f8" in to_squares:
                    move = chess.Move.from_uci(from_ + "g8")
        else:
            # this is an unknown move
            pass
        # TODO: if pawn lands on last file => promote it to queen
        if move is not None:
            if move in board.legal_moves:
                board.push(move)
                print(move)
            else:
                print("Illegal move")
        else:
            print("Unknown move")
        board_state = new_board_state

    for idx, color in new_board_state.items():
        text_pos = np.array((sorted_coordinates[idx][0], sorted_coordinates[idx][1] + 20), np.uint)
        color = (0, 0, 0) if color else (255, 255, 255)
        cv2.putText(bgr_image, str(board.piece_at(image_to_board(idx))), text_pos, font, font_scale, color, thickness)

    cv2.imshow("Live Stream", bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop = True
        break

capture.release()
cv2.destroyAllWindows()

# %%

