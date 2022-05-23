import cv2
import dlib
import math
from math import hypot

def midpoint(point1, point2):
    return (point1.x + point2.x) / 2, (point1.y + point2.y) / 2


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def eye_ratio(eye_points, facial_landmarks):
    # loading all the required points
    corner_left = (facial_landmarks.part(eye_points[0]).x,
                   facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x,
                    facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]),facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]),facial_landmarks.part(eye_points[4]))



    hor_line_length = hypot((corner_left[0]-corner_right[0]),(corner_left[1]-corner_right[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))


    ratio = hor_line_length / ver_line_length



    return ratio


# livestream from the webcam
cap = cv2.VideoCapture(0)


cv2.namedWindow('Drowsiness_detection')

# Face detection with dlib
detector = dlib.get_frontal_face_detector()

# Detecting Eyes using landmarks in dlib
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# these landmarks are for the eye
left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]

count = 0
font = cv2.FONT_HERSHEY_TRIPLEX

while True:

    retval, frame = cap.read()

    if not retval:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # converting image to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # detecting faces in the frame
    faces, _, _ = detector.run(image=frame, upsample_num_times=0,
                               adjust_threshold=0.0)

    # Detecting Eyes using landmarks in dlib
    for face in faces:

        landmarks = predictor(frame, face)

        # -----Step 5: Calculating blink ratio for one eye-----
        left_eye_ratio = eye_ratio(left_eye_landmarks, landmarks)
        right_eye_ratio = eye_ratio(right_eye_landmarks, landmarks)
        eye_open_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if eye_open_ratio > 4.0 or eye_open_ratio > 4.30:
            count += 1
        else:
            count = 0


        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        if count > 10:
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
            cv2.putText(frame, "Watch The Road ", (x, y - 5), font, 0.5, (0, 0, 255))

        else:
            cv2.rectangle(frame, (x,y), (x1,y1), (0, 255, 0), 2)

    cv2.imshow('Drowsiness_detection', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

# releasing the VideoCapture object
cap.release()
cv2.destroyAllWindows()