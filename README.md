# ✊✋✌ Hand Gesture Controlled Rock--Paper--Scissors Game

An interactive **Rock--Paper--Scissors game** controlled entirely using
your **hand gestures** through the webcam.
Built with **Python**, **OpenCV**, and **MediaPipe**, this game detects
your hand shape and plays against the computer in real time.

------------------------------------------------------------------------

## ✨ Features

-   🖐️ **Hand gesture recognition** for Rock, Paper, and Scissors
-   🤖 **Computer opponent** with randomized choices
-   🎥 Real-time webcam display with detected hand landmarks
-   🏆 Score tracking for both User and Computer
-   🖼️ Display of your **hand region** (cropped preview)
-   ⏱️ Automatic round timer
-   📢 Clear on-screen feedback after every round

------------------------------------------------------------------------

## 📁 File Structure

-   `main.py` --- Complete game code including gesture detection, timer
    logic, scoring system, and UI overlay\
-   `README.md` --- Documentation for this project

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   Python
-   OpenCV
-   MediaPipe
-   NumPy

------------------------------------------------------------------------

## 🚀 Getting Started

``` bash
1. Clone or download the project folder

2. Install dependencies : pip install opencv-python mediapipe numpy

3. Run the game : python main.py
```

------------------------------------------------------------------------

## 🎮 How to Play

-   🖐 Show your hand in front of the camera
-   ✊ Make a **fist** → Rock
-   ✋ Keep all fingers straight → Paper
-   ✌ Extend **index + middle fingers** → Scissors
-   Each round lasts **3 seconds**
-   After each round, results will be displayed

------------------------------------------------------------------------

## 📌 Controls

  Action   Gesture/Key
  -------- --------------------------------------
  Play     Show Rock / Paper / Scissors gesture
  Quit     Press `Q`

------------------------------------------------------------------------

## 📊 Game Logic

1.  User gesture is detected using **MediaPipe hand landmarks**
2.  Computer generates a random choice
3.  Winner is determined using standard game rules
4.  Scores update automatically

------------------------------------------------------------------------

## 🧠 Gesture Detection Logic

-   **Rock** → All fingers folded
-   **Paper** → All fingers extended
-   **Scissors** → Only index & middle extended
-   Any incorrect shape → "Unclear" or "No Input"

------------------------------------------------------------------------

## 🏆 Winning

The score is displayed on the screen at all times:

    User: X | Computer: Y

Play as many rounds as you like!

------------------------------------------------------------------------

## 📱 Responsive Experience

-   Shows your **cropped hand preview** at the corner
-   Real-time gesture visualization using **MediaPipe landmarks**
-   Smooth FPS and low-latency interaction

------------------------------------------------------------------------

## ✔️ Requirements

-   Webcam
-   Python 3.7+
-   Good lighting for accurate gesture detection

------------------------------------------------------------------------

## 💡 Future Improvements

-   Add sound effects
-   Add UI menus
-   Add animation for round transitions
-   Add multiplayer mode

------------------------------------------------------------------------

## 📜 License

This project is free to use, modify, and distribute.

Enjoy playing Rock--Paper--Scissors using just your hand gestures! ✊✋✌

---
