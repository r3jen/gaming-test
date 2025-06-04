import cv2

class Player:
    def __init__(self):
        self.position = (0, 0)
        self.image = cv2.imread("player.png", cv2.IMREAD_UNCHANGED)  # emoji player PNG
        self.size = 50
        self.health = 1

    def update(self, hand_position):
        self.position = hand_position

    def draw(self, frame):
        self.overlay_image(frame, self.image, self.position)

    def overlay_image(self, background, overlay, position):
        x, y = position
        overlay = cv2.resize(overlay, (self.size, self.size))
        y1, y2 = y, y + overlay.shape[0]
        x1, x2 = x, x + overlay.shape[1]

        if y2 > background.shape[0] or x2 > background.shape[1]:
            return

        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3] / 255.0
            for c in range(3):
                background[y1:y2, x1:x2, c] = (
                    alpha * overlay[:, :, c] + (1 - alpha) * background[y1:y2, x1:x2, c]
                )
