import cv2
from player import Player
from boss import Boss
from hand_tracker import HandTracker
import numpy as np
from collections import deque

class GameEngine:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.WIDTH, self.HEIGHT = 640, 480
        self.cap.set(3, self.WIDTH)
        self.cap.set(4, self.HEIGHT)

        self.player = Player()
        self.boss = Boss(self.WIDTH, self.HEIGHT)
        self.hand_tracker = HandTracker()
        self.history = deque(maxlen=15)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def run(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            pos = self.hand_tracker.get_index_finger(frame)

            if pos:
                self.player.update(pos)
                self.history.append(np.array(pos))

            # Prediksi pergerakan dan gerakkan boss
            if len(self.history) > 1:
                predicted = self.history[-1] + (self.history[-1] - self.history[0]) / len(self.history)
                self.boss.move_toward(predicted)

            self.player.draw(frame)
            self.boss.draw(frame)

            # Deteksi tabrakan
            if pos:
                player_pos = np.array(pos)
                boss_pos = self.boss.position
                dist = np.linalg.norm(player_pos - boss_pos)

                player_radius = self.player.size // 2
                boss_radius = self.boss.size // 2

                if dist < (player_radius + boss_radius):

                    self.player.health -= 1
                    self.boss.position = np.array([300, 100], dtype=np.float64)

            # Tampilkan HP
            cv2.putText(frame, f"HP: {self.player.health}", (10, 30), self.font, 1, (255, 255, 255), 2)

            # GAME OVER
            if self.player.health <= 0:
                while True:
                    game_over_text = "GAME OVER - Press 'R' to Restart or 'Q' to Quit"
                    cv2.putText(frame, game_over_text, (40, 240), self.font, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Game", frame)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('r'):
                        self.__init__()  # Reset
                        self.run()
                        return
                    elif key == ord('q'):
                        self.cap.release()
                        cv2.destroyAllWindows()
                        return
                break

            cv2.imshow("Game", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
