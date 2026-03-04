import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import mediapipe as mp

print("Starting gesture recognition...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model architecture (same as training)
class GestureCNN(nn.Module):
    def __init__(self):
        super(GestureCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64*14*14,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


print("Loading model...")

model = GestureCNN().to(device)
model.load_state_dict(torch.load("models/gesture_model.pth", map_location=device))
model.eval()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

# MediaPipe setup
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

print("Opening webcam...")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = frame.shape

            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            x_min = min(x_list) - 20
            x_max = max(x_list) + 20
            y_min = min(y_list) - 20
            y_max = max(y_list) + 20

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:

                img = transform(hand_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(img)
                    _, predicted = torch.max(outputs,1)

                gesture = predicted.item()

                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (x_min, y_min-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

                cv2.rectangle(
                    frame,
                    (x_min,y_min),
                    (x_max,y_max),
                    (0,255,0),
                    2
                )

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()