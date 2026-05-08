import math
import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3
import threading
import time

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
MODEL_PATH = r'C:\Users\mutentiza\Videos\Sign-Language-To-Text-and-Speech-Conversion\cnn8grps_rad1_model.h5'
WHITE_IMG_PATH = r'C:\Users\mutentiza\Videos\Sign-Language-To-Text-and-Speech-Conversion\white.jpg'
TASK_PATH = r'C:\Users\mutentiza\OneDrive\Documents\sign langiage model\hand_landmarker.task'

# ==========================================
# HAND DETECTOR WRAPPER
# ==========================================
class HandDetector:
    def __init__(self, maxHands=1):
        self.maxHands = maxHands
        base_options = python.BaseOptions(model_asset_path=TASK_PATH)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=self.maxHands)
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def findHands(self, img, draw=True):
        if img is None or img.size == 0: return []
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            result = self.landmarker.detect(mp_image)
        except: return []

        allHands = []
        h, w, c = img.shape
        if result and result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                myHand = {}
                mylmList = []
                xList, yList = [], []
                for lm in hand_landmarks:
                    px, py = int(lm.x * w), int(lm.y * h)
                    mylmList.append([px, py, lm.z])
                    xList.append(px); yList.append(py)
                
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                allHands.append(myHand)
        return allHands

# ==========================================
# VOICE ENGINE
# ==========================================
def speak(text):
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except: pass
    threading.Thread(target=_speak, daemon=True).start()

# ==========================================
# DISTANCE HELPER
# ==========================================
def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    print("Loading Model...")
    model = load_model(MODEL_PATH)
    hd = HandDetector(maxHands=1)
    hd2 = HandDetector(maxHands=1)
    
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("ERROR: Could not open webcam. Make sure no other app is using it!")
        return
    
    offset = 29
    
    # State Management
    current_sentence = []
    current_word = ""
    last_added_letter = ""
    voting_buffer = []
    VOTING_SIZE = 15 # Wait for 15 frames of same letter
    cooldown = 0
    
    print("AI Sign-to-Speech UI Started. Window should appear now...")
    
    while True:
        ret, frame = capture.read()
        if not ret: 
            print("ERROR: Lost connection to camera. Exiting...")
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        hands = hd.findHands(frame)
        
        predicted_char = ""
        
        if hands:
            hand = hands[0]
            bx, by, bw, bh = hand['bbox']
            
            # Crop and prepare for second hand detection
            try:
                crop = frame[max(0, by-offset):min(h, by+bh+offset), max(0, bx-offset):min(w, bx+bw+offset)]
                white = np.ones((400, 400, 3), np.uint8) * 255
                
                handz = hd2.findHands(crop)
                if handz:
                    pts = handz[0]['lmList']
                    os = ((400 - bw) // 2) - 15
                    os1 = ((400 - bh) // 2) - 15
                    
                    # Draw Skeleton on White Canvas
                    connections = [(0,1,2,3,4), (5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]
                    for path in connections:
                        for i in range(len(path)-1):
                            p1, p2 = pts[path[i]], pts[path[i+1]]
                            cv2.line(white, (p1[0]+os, p1[1]+os1), (p2[0]+os, p2[1]+os1), (0, 255, 0), 3)
                    
                    # Cross-finger connections
                    cv2.line(white, (pts[5][0]+os, pts[5][1]+os1), (pts[9][0]+os, pts[9][1]+os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[9][0]+os, pts[9][1]+os1), (pts[13][0]+os, pts[13][1]+os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[13][0]+os, pts[13][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0]+os, pts[0][1]+os1), (pts[5][0]+os, pts[5][1]+os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0]+os, pts[0][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (0, 255, 0), 3)

                    for pt in pts:
                        cv2.circle(white, (pt[0]+os, pt[1]+os1), 2, (0, 0, 255), -1)

                    # Group-based logic from original script
                    white_input = white.reshape(1, 400, 400, 3)
                    prob = np.array(model.predict(white_input, verbose=0)[0], dtype='float32')
                    ch1 = np.argmax(prob, axis=0)
                    prob[ch1] = 0
                    ch2 = np.argmax(prob, axis=0)
                    
                    pl = [ch1, ch2]

                    # --- ORIGINAL CLASSIFICATION LOGIC START ---
                    # [0->aemnst][1->bfdiuvwkr][2->co][3->gh][4->l][5->pqz][6->x][7->yj]
                    
                    # Group refinements
                    if pl in [[5,2],[5,3],[3,5],[3,6],[3,0],[3,2],[6,4],[6,1],[6,2],[6,6],[6,7],[6,0],[6,5],[4,1],[1,0],[1,1],[6,3],[1,6],[5,6],[5,1],[4,5],[1,4],[1,5],[2,0],[2,6],[4,6],[1,0],[5,7],[1,6],[6,1],[7,6],[2,5],[7,1],[5,4],[7,0],[7,5],[7,2]]:
                        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]): ch1=0
                    if pl in [[2,2],[2,1]]:
                        if (pts[5][0] < pts[4][0] ): ch1=0
                    if pl in [[0,0],[0,6],[0,2],[0,5],[0,1],[0,7],[5,2],[7,6],[7,1]]:
                        if (pts[0][0]>pts[8][0] and pts[0][0]>pts[4][0] and pts[0][0]>pts[12][0] and pts[0][0]>pts[16][0] and pts[0][0]>pts[20][0]) and pts[5][0] > pts[4][0]: ch1=2
                    if pl in [[6,0],[6,6],[6,2]]:
                        if distance(pts[8],pts[16])<52: ch1 = 2
                    if pl in [[1,4],[1,5],[1,6],[1,3],[1,0]]:
                        if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1]<pts[20][1] and pts[0][0]<pts[8][0] and pts[0][0]<pts[12][0] and pts[0][0]<pts[16][0] and pts[0][0]<pts[20][0]: ch1 = 3
                    if pl in [[4,6],[4,1],[4,5],[4,3],[4,7]]:
                        if pts[4][0]>pts[0][0]: ch1=3
                    if pl in [[5, 3],[5,0],[5,7], [5, 4], [5, 2],[5,1],[5,5]]:
                        if pts[2][1]+15<pts[16][1]: ch1 = 3
                    if pl in [[6, 4], [6, 1], [6, 2]]:
                        if distance(pts[4],pts[11])>55: ch1 = 4
                    if pl in [[1, 4], [1, 6],[1,1]]:
                        if (distance(pts[4], pts[11]) > 50) and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]): ch1 = 4
                    if pl in [[3, 6], [3, 4]]:
                        if (pts[4][0]<pts[0][0]): ch1 = 4
                    if pl in [[2, 2], [2, 5],[2,4]]:
                        if (pts[1][0] < pts[12][0]): ch1 = 4
                    if pl in [[3, 6],[3,5],[3,4]]:
                        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]) and pts[4][1]>pts[10][1]: ch1 = 5
                    if pl in [[3,2],[3,1],[3,6]]:
                        if pts[4][1]+17>pts[8][1] and pts[4][1]+17>pts[12][1] and pts[4][1]+17>pts[16][1] and pts[4][1]+17>pts[20][1]: ch1 = 5
                    if pl in [[4,4],[4,5],[4,2],[7,5],[7,6],[7,0]]:
                        if pts[4][0]>pts[0][0]: ch1 = 5
                    if pl in [[0, 2],[0,6],[0,1],[0,5],[0,0],[0,7],[0,4],[0,3],[2,7]]:
                        if pts[0][0]<pts[8][0]  and  pts[0][0]<pts[12][0]  and pts[0][0]<pts[16][0]  and pts[0][0]<pts[20][0]: ch1 = 5
                    if pl in [[5, 7],[5,2],[5,6]]:
                        if pts[3][0]<pts[0][0]: ch1 = 7
                    if pl in [[4, 6],[4,2],[4,4],[4,1],[4,5],[4,7]]:
                        if pts[6][1] < pts[8][1]: ch1 = 7
                    if pl in [[6, 7],[0,7],[0,1],[0,0],[6,4],[6,6] ,[6,5],[6,1]]:
                        if pts[18][1] > pts[20][1]: ch1 = 7
                    if pl in [[0,4],[0,2],[0,3],[0,1],[0,6]]:
                        if pts[5][0]>pts[16][0]: ch1 = 6
                    if pl in [[7, 2]]:
                        if pts[18][1] < pts[20][1]: ch1 = 6
                    if pl in [[2, 1],[2,2],[2,6],[2,7],[2,0]]:
                        if distance(pts[8],pts[16])>50: ch1 = 6
                    if pl in [[4, 6],[4,2],[4,1],[4,4]]:
                        if distance(pts[4], pts[11]) < 60: ch1 = 6
                    if pl in [[1,4],[1,6],[1,0],[1,2]]:
                        if pts[5][0] - pts[4][0] - 15 > 0: ch1 = 6
                    if pl in [[5,0],[5,1],[5,4],[5,5],[5,6],[6,1],[7,6],[0,2],[7,1],[7,4],[6,6],[7,2],[5,0],[6,3],[6,4],[7,5],[7,2]]:
                        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]): ch1 = 1
                    if pl in [[6, 1],[6,0],[0,3],[6,4],[2,2], [0,6],[6,2],[7, 6],[4,6],[4,1],[4,2], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]:
                        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]): ch1 = 1
                    if pl in [[5,0],[3,4],[3,0],[3,1],[3,5],[5,5],[5,4],[5,1],[7,6]]:
                        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0]<pts[0][0]) and pts[4][1]>pts[14][1]): ch1 = 1
                    if pl in [[5,4],[5,5],[5,1],[0,3],[0,7],[5,0],[0,2],[6,2],[7, 5], [7, 1], [7, 6], [7, 7]]:
                        if ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])): ch1 = 1
                    if pl in [[5,5],[5,0],[5,4],[5,1],[4,6],[4,1],[7,6],[3,0],[3,5]]:
                        if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])) and pts[4][1]>pts[14][1]: ch1 = 1
                    if pl in [[3,5],[3,0],[3,6],[5,1],[4,1],[2,0],[5,0],[5,5]]:
                        if not(pts[0][0]+13 < pts[8][0] and pts[0][0]+13 < pts[12][0] and pts[0][0]+13 < pts[16][0]  and pts[0][0]+13 < pts[20][0]) and not(pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0]  and pts[0][0] > pts[20][0]) and distance(pts[4], pts[11]) < 50: ch1 = 1

                    # Final mapping
                    res = ""
                    if ch1 == 0:
                        res = 'S'
                        if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]: res = 'A'
                        if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]: res = 'T'
                        if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]: res = 'E'
                        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]: res = 'M'
                        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]: res = 'N'
                    elif ch1 == 2:
                        res = 'C' if distance(pts[12], pts[4]) > 42 else 'O'
                    elif ch1 == 3:
                        res = 'G' if distance(pts[8], pts[12]) > 72 else 'H'
                    elif ch1 == 7:
                        res = 'Y' if distance(pts[8], pts[4]) > 42 else 'J'
                    elif ch1 == 4: res = 'L'
                    elif ch1 == 6: res = 'X'
                    elif ch1 == 5:
                        if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
                            res = 'Z' if pts[8][1] < pts[5][1] else 'Q'
                        else: res = 'P'
                    elif ch1 == 1:
                        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] >pts[20][1]): res = 'B'
                        elif (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]): res = 'D'
                        elif (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]): res = 'F'
                        elif (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]): res = 'I'
                        elif (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]): res = 'W'
                        elif (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1]<pts[9][1]: res = 'K'
                        elif ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]): res = 'U'
                        elif ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] >pts[9][1]): res = 'V'
                        elif (pts[8][0] > pts[12][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]): res = 'R'

                    # Special commands
                    if ch1 in [1, 'E', 'S', 'X', 'Y', 'B']:
                        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]): res = 'Space'
                    if res in ['E', 'Y', 'B']:
                        if (pts[4][0] < pts[5][0] ): res = 'Next'
                    if res in ['Next', 'B', 'C', 'H', 'F']:
                        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[4][1]<pts[8][1] and pts[4][1]<pts[12][1] and pts[4][1]<pts[16][1] and pts[4][1]<pts[20][1]: res = 'Backspace'
                    
                    predicted_char = res

                    voting_buffer.append(predicted_char)
                    if len(voting_buffer) > VOTING_SIZE: voting_buffer.pop(0)
                    
                    # Stabilization
                    from collections import Counter
                    voting_res = Counter(voting_buffer).most_common(1)[0]
                    most_common, count = voting_res[0], voting_res[1]
                    
                    if count >= 10 and cooldown == 0:
                        if most_common == 'Next':
                            if current_word:
                                current_sentence.append(current_word)
                                speak(current_word)
                                current_word = ""
                            cooldown = 25
                        elif most_common == 'Backspace':
                            if current_word: current_word = current_word[:-1]
                            cooldown = 25
                        elif most_common == 'Space':
                            if current_word:
                                current_sentence.append(current_word)
                                current_word = ""
                            cooldown = 25
                        elif most_common not in ['', 'Next', 'Backspace', 'Space']:
                            if most_common != last_added_letter:
                                current_word += most_common
                                last_added_letter = most_common
                                cooldown = 20
                                voting_buffer = [] # Clear buffer after adding
                    # --- ORIGINAL CLASSIFICATION LOGIC END ---
            except Exception as e: pass
        else:
            voting_buffer = []
            last_added_letter = ""

        if cooldown > 0: cooldown -= 1

        # ==========================================
        # DRAW PREMIUM HUD
        # ==========================================
        # Top Panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (w-20, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, f"LIVE: {predicted_char}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(frame, f"WORD: {current_word}", (w//2, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Bottom Panel
        cv2.rectangle(frame, (20, h-80), (w-20, h-20), (50, 50, 50), -1)
        sentence_str = " ".join(current_sentence)
        cv2.putText(frame, f"SENTENCE: {sentence_str}", (40, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)

        cv2.imshow("Sign-to-Speech AI", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        if key == ord('s'): # Manual Speak Trigger
            if current_word:
                speak(current_word)
                current_sentence.append(current_word)
                current_word = ""

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
