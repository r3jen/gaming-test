# === FILE: ai_boss_battle.py ===
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import random

# --- Hand Tracking Init ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Game Config ---
WIDTH, HEIGHT = 640, 480
player_radius = 20
boss_radius = 40
max_history = 15  # for movement prediction

# --- Init Camera ---
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# --- State ---
player_history = deque(maxlen=max_history)
boss_pos = np.array([random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100)])
boss_velocity = np.array([0, 0])
player_health = 5
boss_health = 10

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    player_pos = None

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmark.landmark[8]
            x, y = int(lm.x * WIDTH), int(lm.y * HEIGHT)
            player_pos = np.array([x, y])
            player_history.append(player_pos)

            # Draw player
            cv2.circle(frame, tuple(player_pos), player_radius, (0, 255, 0), cv2.FILLED)

    # --- Predict next move (basic linear extrapolation) ---
    if len(player_history) >= 2:
        diff = player_history[-1] - player_history[0]
        predicted_pos = player_history[-1] + diff // len(player_history)
        predicted_pos = np.clip(predicted_pos, [0, 0], [WIDTH, HEIGHT])
    else:
        predicted_pos = np.array([WIDTH // 2, HEIGHT // 2])

    # --- Boss attack logic (move toward predicted position) ---
    direction = predicted_pos - boss_pos
    if np.linalg.norm(direction) > 1:
        boss_velocity = direction / np.linalg.norm(direction) * 3
        boss_pos += boss_velocity

    # --- Draw boss ---
    cv2.circle(frame, tuple(boss_pos.astype(int)), boss_radius, (0, 0, 255), cv2.FILLED)

    # --- Check collision (boss hits player) ---
    if player_pos is not None:
        dist = np.linalg.norm(boss_pos - player_pos)
        if dist < (player_radius + boss_radius):
            player_health -= 1
            boss_pos = np.array([random.randint(100, WIDTH - 100), random.randint(100, HEIGHT - 100)])
            cv2.putText(frame, "Hit!", (WIDTH // 2 - 50, HEIGHT // 2), font, 1, (0, 0, 255), 3)

    # --- Draw Health Info ---
    cv2.putText(frame, f"Player HP: {player_health}", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Boss HP: {boss_health}", (WIDTH - 180, 30), font, 0.7, (255, 255, 255), 2)

    # --- Game Over ---
    if player_health <= 0:
        cv2.putText(frame, "GAME OVER", (WIDTH // 2 - 100, HEIGHT // 2), font, 1.5, (0, 0, 255), 4)
        cv2.imshow("AI Boss Battle", frame)
        cv2.waitKey(3000)
        break

    cv2.imshow("AI Boss Battle", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
