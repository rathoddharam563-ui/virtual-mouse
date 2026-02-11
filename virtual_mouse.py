import cv2
import mediapipe as mp
import pyautogui
import math
import time
from playsound import playsound
import threading

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

# ========= CONFIG =========
smooth = 10
prev_x, prev_y = 0, 0
cooldown = 0.7
last_action = 0
pinch_dist = 35
# ==========================

# 🔊 sound helper (non-blocking)
def play(sound):
    threading.Thread(target=playsound, args=(sound,), daemon=True).start()

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def dist(a, b):
    return math.hypot(b[0]-a[0], b[1]-a[1])

def finger_up(tip, pip):
    return tip.y < pip.y

def glass_panel(frame, x1, y1, x2, y2):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    icon = "❓"
    color = (120, 120, 120)

    # ===== Glass UI =====
    glass_panel(frame, 20, 20, 180, 90)
    cv2.putText(frame, "creathe by:-Dharam Rathod", (35, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    glass_panel(frame, w-180, 20, w-20, 150)

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark

        index = (int(lm[8].x*w), int(lm[8].y*h))
        middle = (int(lm[12].x*w), int(lm[12].y*h))
        thumb = (int(lm[4].x*w), int(lm[4].y*h))

        index_up = finger_up(lm[8], lm[6])
        middle_up = finger_up(lm[12], lm[10])
        ring_up = finger_up(lm[16], lm[14])
        pinky_up = finger_up(lm[20], lm[18])

        now = time.time()

        # ===== PRIORITY GESTURES =====

        # 👆 LEFT CLICK
        if dist(index, thumb) < pinch_dist and now - last_action > cooldown:
            pyautogui.click()
            play("click.wav")
            last_action = now
            icon = "👆"
            color = (0, 200, 0)

        # ✊ RIGHT CLICK
        elif not index_up and not middle_up and not ring_up and not pinky_up and now - last_action > cooldown:
            pyautogui.rightClick()
            play("right.wav")
            last_action = now
            icon = "✊"
            color = (180, 0, 180)

        # 🔄 SCROLL
        elif index_up and middle_up:
            pyautogui.scroll(40 if index[1] < middle[1] else -40)
            icon = "🔄"
            color = (0, 0, 255)

        # 🖱 MOVE
        elif index_up and not middle_up:
            x = int(index[0] * screen_w / w)
            y = int(index[1] * screen_h / h)
            cx = prev_x + (x - prev_x) / smooth
            cy = prev_y + (y - prev_y) / smooth
            pyautogui.moveTo(cx, cy)
            prev_x, prev_y = cx, cy
            icon = "🖱"
            color = (255, 120, 0)

        mp_draw.draw_landmarks(frame,
                               result.multi_hand_landmarks[0],
                               mp_hands.HAND_CONNECTIONS)

    # ===== Gesture Icon =====
    cv2.rectangle(frame, (w-180, 20), (w-20, 150), color, -1)
    cv2.putText(frame, icon, (w-135, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8)

    cv2.imshow("AI Virtual Mouse PRO", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
