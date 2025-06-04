import numpy as np
import random
import cv2

class Boss:
    def __init__(self, width, height):
        self.position = np.array([random.randint(100, width - 100), random.randint(100, height - 100)], dtype=np.float64)
        self.image = cv2.imread("boss.png", cv2.IMREAD_UNCHANGED)  # emoji boss PNG (with alpha)
        self.size = 60

    def move_toward(self, target):
        direction = target - self.position
        if np.linalg.norm(direction) > 1:
            speed = 7  # kamu bisa ubah jadi 8, 10, dsb untuk makin cepat
            velocity = direction / np.linalg.norm(direction) * speed
            self.position += velocity

    def draw(self, frame):
        self.overlay_image(frame, self.image, self.position.astype(int))

    def overlay_image(self, background, overlay, position):
        x, y = position
        h, w = overlay.shape[:2]
        overlay = cv2.resize(overlay, (self.size, self.size))
        y1, y2 = y, y + overlay.shape[0]
        x1, x2 = x, x + overlay.shape[1]

        if y2 > background.shape[0] or x2 > background.shape[1]:
            return  # avoid overflow

        if overlay.shape[2] == 4:  # has alpha channel
            alpha = overlay[:, :, 3] / 255.0
            for c in range(3):
                background[y1:y2, x1:x2, c] = (
                    alpha * overlay[:, :, c] + (1 - alpha) * background[y1:y2, x1:x2, c]
                )
