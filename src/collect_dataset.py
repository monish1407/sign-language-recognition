import cv2
import mediapipe as mp
import os

gesture = input("Enter gesture name: ")

save_path = f"dataset/{gesture}"
os.makedirs(save_path, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

count = 0
max_images = 200
hand_img = None

print("Press SPACE to capture image")
print("Press Q to quit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    hand_detected = False

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape

            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            x_min = max(min(x_list) - 20, 0)
            x_max = min(max(x_list) + 20, w)
            y_min = max(min(y_list) - 20, 0)
            y_max = min(max(y_list) + 20, h)

            hand_img = frame[y_min:y_max, x_min:x_max]
            hand_detected = True

            if hand_img.size != 0:
                cv2.imshow("Hand Crop", hand_img)

    if not hand_detected:
        cv2.putText(frame, "Show your hand", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):

        if hand_img is not None:

            file_path = f"{save_path}/{count}.jpg"
            cv2.imwrite(file_path, hand_img)

            print("Saved:", file_path)

            count += 1

            if count >= max_images:
                print("Collected 200 images.")
                break

        else:
            print("Hand not detected!")

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()