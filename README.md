\# Real-Time Hand Gesture Recognition



This project implements a real-time hand gesture recognition system using deep learning and computer vision. The system detects hand gestures from a webcam feed and classifies them using a trained CNN model.



\## Features



\- Real-time gesture detection using webcam

\- CNN-based gesture classification using PyTorch

\- Hand detection using MediaPipe

\- OpenCV for real-time video processing



\## Tech Stack



\- Python

\- PyTorch

\- OpenCV

\- MediaPipe



\## Project Structure



```

sign-language-recognition

│

├── src

│   ├── inference.py

│   └── collect\_dataset.py

│

├── models

│   └── gesture\_model.pth

│

├── dataset

│

├── requirements.txt

└── README.md

```



\## Installation



Install required libraries:



```

pip install -r requirements.txt

```



\## Run the Project



Run the real-time gesture recognition system:



```

python src/inference.py

```



The webcam will open and detect hand gestures in real time.



\## Future Improvements



\- Improve model accuracy with larger datasets

\- Add more gesture classes

\- Deploy as a web application

