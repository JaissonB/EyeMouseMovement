import cv2
import numpy as np
import pyautogui
import dlib
import winsound

face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)
screen_w, screen_h = pyautogui.size()
prev_left_eye_center = (0,0)
prev_right_eye_center = None
num_frames_for_smoothing = 20
eye_centers_buffer = []
blinking_buffer = []
last_moves = []
mouse_move_speed = 15

predictor_path = "./shape_predictor_68_face_landmarks.dat"
detector_dlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

threshold_value = 128

toggle_mouse_move = False
scroll_screen = False
left_eye_frame_blinking = False
zoom_in = False

def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame

def binarize_image(img):
    ###############################
    cv2.namedWindow("Normal EYE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Normal EYE", 300, 125)
    cv2.imshow("Normal EYE", img)

    cv2.namedWindow("Binary EYE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Binary EYE", 300, 125)

    cv2.namedWindow("Gray EYE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gray EYE", 300, 125)

    cv2.namedWindow("Histogram EYE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Histogram EYE", 300, 125)
    
    cv2.namedWindow("Vignette EYE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vignette EYE", 300, 125)
    ###############################

    global contrast, bright, threshold_value

    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray EYE", gray_frame)
    gray_frame = cv2.equalizeHist(gray_frame)
    cv2.imshow("Histogram EYE", gray_frame)
    gray_frame = apply_vignette(gray_frame)
    cv2.imshow("Vignette EYE", gray_frame)

    _, img = cv2.threshold(gray_frame, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow("Binary EYE", img)

    return img

def detect_eyes(img, eye_cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_frame, 1.3, 5)
    width = np.size(img, 1)
    height = np.size(img, 0)
    left_eye = None
    right_eye = None
    left_eye_coords = ()
    right_eye_coords = ()
    for (x, y, w, h) in eyes:
        if y > height / 2:
            continue
        eyecenter = x + w / 2
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
            left_eye_coords = (x, y, w, h)
        else:
            right_eye = img[y:y + h, x:x + w]
            right_eye_coords = (x, y, w, h)
    return left_eye, right_eye, left_eye_coords, right_eye_coords

def nothing(x):
    pass

def apply_vignette(img, focus_factor=1):
    height, width = img.shape[:2]

    if len(img.shape) == 3:
        gradient = np.ones((height, width, img.shape[2]), dtype=np.float32)
    else:
        gradient = np.ones((height, width), dtype=np.float32)

    center_x, center_y = width // 2, height // 2
    max_radius = np.sqrt(center_x**2 + center_y**2)

    for y in range(height):
        for x in range(width):
            radius = np.sqrt((center_x - x)**2 + (center_y - y)**2)
            gradient_value = (radius / max_radius) * focus_factor
            gradient[y, x] = 1 - min(gradient_value, 1)

    vignette_img = cv2.multiply(img.astype(np.float32), gradient)
    vignette_img = np.clip(vignette_img, 0, 255).astype(img.dtype)

    return vignette_img

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    lower_bound_h = int(height * (3/4))
    img = img[eyebrow_h:lower_bound_h, 0:width]

    return img

def analyze_eye_direction(binarized_eye_img, size_threshold=57, margin_ratio=0.1):
    ###############################
    cv2.namedWindow("Morfologia", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Morfologia", 300, 125)
    
    cv2.namedWindow("LEFT side", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LEFT side", 300, 125)
    
    cv2.namedWindow("RIGHT side", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RIGHT side", 300, 125)
    
    cv2.namedWindow("Lower Center", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lower Center", 300, 125)
    ###############################
    global last_moves, mouse_move_speed, zoom_in

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_img = cv2.morphologyEx(binarized_eye_img, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Morfologia", cleaned_img)
    white_pixels_total = cv2.countNonZero(cleaned_img)

    margin = white_pixels_total * margin_ratio

    height, width = cleaned_img.shape
    left_half = cleaned_img[:, :width // 2]
    right_half = cleaned_img[:, width // 2:]
    cv2.imshow("LEFT side", left_half)
    cv2.imshow("RIGHT side", right_half)

    white_pixels_left = cv2.countNonZero(left_half)
    white_pixels_right = cv2.countNonZero(right_half)

    vertical_start = int(height * 0.6)
    vertical_end = height
    horizontal_start = int(width * 0.3)
    horizontal_end = int(width * 0.7)

    lower_center = cleaned_img[vertical_start:vertical_end, horizontal_start:horizontal_end]
    cv2.imshow("Lower Center", lower_center)
    white_pixels_lower_center = cv2.countNonZero(lower_center)

    if white_pixels_lower_center < 5:
        direction = 'down'
    elif white_pixels_lower_center < 30:
        direction = 'stop'
    elif white_pixels_lower_center > size_threshold:
        direction = 'up'
    elif white_pixels_left + margin < white_pixels_right:
        direction = 'left'
    elif white_pixels_right + margin < white_pixels_left:
        direction = 'right'
    elif white_pixels_lower_center > size_threshold//2:
        direction = 'up'
    else: 
        direction = 'stop'

    if not zoom_in:
        if len(last_moves) == 0 or last_moves[len(last_moves)-1] == direction:
            last_moves.append(direction)
            mouse_move_speed = mouse_move_speed + int(len(last_moves)//3)
        else:
            last_moves = []
            mouse_move_speed = 15

    return direction

def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])

    ear = (A + B) / (2.0 * C)
    return ear

def detect_blinking(landmarks, frame, ear_threshold=0.21):
    global zoom_in, toggle_mouse_move, blinking_buffer, left_eye_frame_blinking, scroll_screen, mouse_move_speed
    LEFT_EYE_INDICES = list(range(36, 42))
    RIGHT_EYE_INDICES = list(range(42, 48))
    
    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_INDICES])
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_INDICES])

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)

    left_blinking = left_ear < ear_threshold
    right_blinking = right_ear < ear_threshold

    if len(blinking_buffer) > num_frames_for_smoothing:
        blinking_buffer.pop(0)

    if ((len(blinking_buffer) >= num_frames_for_smoothing) and blinking_buffer.count("Left eye blinking") >= num_frames_for_smoothing//2):
        if toggle_mouse_move and not scroll_screen:
            scroll_screen = True
            toggle_mouse_move = not toggle_mouse_move
            pyautogui.click(button='middle')
            winsound.Beep(750, 200)
        else:
            toggle_mouse_move = not toggle_mouse_move
            winsound.Beep(400, 200)
        cv2.putText(frame, "blinking toggled", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        blinking_buffer = []

    if ((len(blinking_buffer) >= num_frames_for_smoothing) and blinking_buffer.count("Right eye blinking") >= num_frames_for_smoothing//2):
        pyautogui.click()
        cv2.putText(frame, "Left click", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        winsound.Beep(350, 100)
        blinking_buffer = []
        scroll_screen = False

    if ((len(blinking_buffer) >= num_frames_for_smoothing) and blinking_buffer.count("Both eyes blinking") >= num_frames_for_smoothing//4*3):
        pyautogui.hotkey('alt', '4')
        zoom_in = not zoom_in
        winsound.Beep(500, 400)
        cv2.putText(frame, "Toggle magnifier", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 3)
        blinking_buffer = []
        scroll_screen = False

    if left_blinking and right_blinking:
        blinking_buffer.append("Both eyes blinking")
        left_eye_frame_blinking = False
        return "Both eyes blinking"
    elif left_blinking:
        left_eye_frame_blinking = True
        blinking_buffer.append("Left eye blinking")
        return "Left eye blinking"
    elif right_blinking:
        blinking_buffer.append("Right eye blinking")
        left_eye_frame_blinking = False
        return "Right eye blinking"
    else:
        blinking_buffer.append("No blink detected")
        left_eye_frame_blinking = False
        return "No blink detected"
    
def moveMouse(right_direction):
    global mouse_move_speed
    actualMousePositiion = pyautogui.position()
    x = actualMousePositiion.x
    y = actualMousePositiion.y
    if not left_eye_frame_blinking:
        if (toggle_mouse_move and right_direction == "right"):
            x = x + mouse_move_speed
            if x > screen_w - 10:
                x = screen_w - 10
        elif (toggle_mouse_move and right_direction == "left"):
            x = x - mouse_move_speed
            if x < 10:
                x = 10
        elif (toggle_mouse_move and right_direction == "up"):
            y = y - mouse_move_speed
            if y < 10:
                y = 10
        elif (toggle_mouse_move and right_direction == "down"):
            y = y + mouse_move_speed
            if y > screen_h - 10:
                y = screen_h - 10

        pyautogui.moveTo(x, y)

def blob_process(img, detector):
    ###############################
    cv2.namedWindow("Normal EYE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Normal EYE", 300, 125)
    cv2.imshow("Normal EYE", img)

    cv2.namedWindow("Binary EYE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Binary EYE", 300, 125)

    cv2.namedWindow("Erode EYE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Erode EYE", 300, 125)

    cv2.namedWindow("Gray EYE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gray EYE", 300, 125)

    cv2.namedWindow("Histogram EYE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Histogram EYE", 300, 125)
    
    cv2.namedWindow("Contrasted EYE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Contrasted EYE", 300, 125)
    ###############################

    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray EYE", gray_frame)

    gray_frame = cv2.equalizeHist(gray_frame)
    cv2.imshow("Histogram EYE", gray_frame)

    alpha = 7
    beta = 249
    contrasted_img = cv2.convertScaleAbs(gray_frame, alpha=alpha, beta=beta)
    cv2.imshow("Contrasted EYE", contrasted_img)

    img = cv2.adaptiveThreshold(contrasted_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow("Binary EYE", img)

    img = cv2.erode(img, None, iterations=2)
    cv2.imshow("Erode EYE", img)

    keypoints = detector.detect(img)

    for keypoint in keypoints:
        x, y = map(int, keypoint.pt)

    return keypoints

def main():
    global threshold_value
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Live window')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector_dlib(gray_frame)

        for face in faces:

            landmarks = predictor(gray_frame, face)
            blink = detect_blinking(landmarks, frame)
            cv2.putText(frame, blink, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            left_eye, right_eye, left_eye_coords, right_eye_coords = detect_eyes(frame, eye_cascade)

            if right_eye is not None:
                right_eye_processed = binarize_image(cut_eyebrows(right_eye))
                right_direction = analyze_eye_direction(right_eye_processed)
                keypoints = blob_process(right_eye, detector)
                eye = cv2.drawKeypoints(right_eye, keypoints, right_eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                moveMouse(right_direction)
                


        cv2.imshow('Live window', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('r'):
            pyautogui.hotkey('ctrl', '4')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
