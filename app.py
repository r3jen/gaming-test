import cv2
import mediapipe as mp
import math
import random

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

snake = []
snake_length = 0
score = 0
food_x, food_y = random.randint(100, 500), random.randint(100, 400)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if not success or img is None:
        print("Gagal membaca kamera. Cek izin atau koneksi kamera.")
        continue

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            x = int(hand.landmark[8].x * w)  # index finger tip
            y = int(hand.landmark[8].y * h)
            snake.append((x, y))

            # keep snake short
            if len(snake) > 20 + score * 2:
                snake.pop(0)

            # draw snake
            for i in range(1, len(snake)):
                cv2.line(img, snake[i-1], snake[i], (255, 255, 255), 10)

            # check collision with food
            if math.hypot(x - food_x, y - food_y) < 20:
                score += 1
                food_x, food_y = random.randint(100, 500), random.randint(100, 400)

    # draw food
    cv2.circle(img, (food_x, food_y), 15, (0, 255, 0), cv2.FILLED)

    # display score
    cv2.putText(img, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Snake Game - Hand Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
