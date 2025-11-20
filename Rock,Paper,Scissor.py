import cv2
import mediapipe as mp
import time
import random
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
font = cv2.FONT_HERSHEY_SIMPLEX

choices = ["Rock", "Paper", "Scissors"]

user_score = 0
computer_score = 0

def detect_gesture(landmarks):
    if not landmarks:
        return None

    lm = landmarks
    finger_states = {
        "thumb": lm[mp_hands.HandLandmark.THUMB_TIP].x < lm[mp_hands.HandLandmark.THUMB_IP].x,
        "index": lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        "middle": lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        "ring": lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        "pinky": lm[mp_hands.HandLandmark.PINKY_TIP].y < lm[mp_hands.HandLandmark.PINKY_PIP].y
    }

    if not any(finger_states.values()):
        return "Rock"
    if all(finger_states.values()):
        return "Paper"
    if finger_states["index"] and finger_states["middle"] and \
       not finger_states["thumb"] and not finger_states["ring"] and not finger_states["pinky"]:
        return "Scissors"

    return "Unclear"

def get_winner(user, computer):
    if user == computer:
        return "Draw"
    elif (user == "Rock" and computer == "Scissors") or \
         (user == "Scissors" and computer == "Paper") or \
         (user == "Paper" and computer == "Rock"):
        return "User"
    else:
        return "Computer"

cap = cv2.VideoCapture(0)
prev_time = time.time()
round_duration = 3
show_result_until = 0
detected_gesture = "None"
result_text = ""
computer_choice = ""
user_feedback_image = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_time = time.time()
    remaining_time = int(round_duration - (current_time - prev_time))

    hand_gesture = "None"
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_gesture = detect_gesture(hand_landmarks.landmark)

            h, w, _ = frame.shape
            points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark])
            bbox = cv2.boundingRect(points)
            x, y, bw, bh = bbox
            user_feedback_image = frame[y:y + bh, x:x + bw].copy() if y > 0 and x > 0 else None

    if current_time - prev_time > round_duration:
        prev_time = current_time
        computer_choice = random.choice(choices)
        detected_gesture = hand_gesture if hand_gesture in choices else "No Input"
        winner = get_winner(detected_gesture, computer_choice) if detected_gesture in choices else "Invalid Input"
        result_text = f"You: {detected_gesture} | Computer: {computer_choice} | Winner: {winner}"

        if winner == "User":
            user_score += 1
        elif winner == "Computer":
            computer_score += 1

        show_result_until = current_time + 3

    if remaining_time >= 0:
        cv2.putText(frame, f"Next round in: {remaining_time}s", (30, 80), font, 1, (0, 0, 255), 2)

    if current_time < show_result_until:
        cv2.putText(frame, result_text, (30, 400), font, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f"User: {user_score} | Computer: {computer_score}", (30, 450), font, 1, (255, 255, 255), 2)

    if user_feedback_image is not None:
        resized_hand = cv2.resize(user_feedback_image, (150, 150))
        h, w, _ = resized_hand.shape
        frame[10:10 + h, frame.shape[1] - w - 10:frame.shape[1] - 10] = resized_hand
        cv2.rectangle(frame, (frame.shape[1] - w - 10, 10), (frame.shape[1] - 10, 10 + h), (0, 255, 0), 2)

    cv2.putText(frame, "Press 'q' to quit", (30, 40), font, 0.7, (0, 0, 255), 2)
    cv2.imshow("Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
