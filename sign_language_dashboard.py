import numpy as np
import math
import cv2
import os, sys
import traceback
import pyttsx3
from keras.models import load_model
from string import ascii_uppercase
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from PIL import Image, ImageTk

# Optional Enchant for suggestions
try:
    import enchant
    ddd = enchant.Dict("en-US")
    HAS_ENCHANT = True
except:
    print("Warning: enchant not found. Suggestions will be disabled.")
    HAS_ENCHANT = False

# ==========================================
# PATHS
# ==========================================
PROJECT_DIR = r'C:\Users\mutentiza\Videos\Sign-Language-To-Text-and-Speech-Conversion'
MODEL_PATH = os.path.join(PROJECT_DIR, 'cnn8grps_rad1_model.h5')
TASK_PATH = r'C:\Users\mutentiza\OneDrive\Documents\sign langiage model\hand_landmarker.task'

# ==========================================
# MODERN HAND DETECTOR
# ==========================================
class HandDetector:
    def __init__(self, maxHands=1):
        self.maxHands = maxHands
        base_options = python.BaseOptions(model_asset_path=TASK_PATH)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=self.maxHands)
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def findHands(self, img, draw=False, flipType=True):
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
                myHand["lmList"] = mylmList
                myHand["bbox"] = (xmin, ymin, xmax - xmin, ymax - ymin)
                allHands.append(myHand)
        return allHands

hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
offset = 29

class Application:
    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        print("Loading AI Model...")
        self.model = load_model(MODEL_PATH)
        
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 150)
        
        self.ct = {'blank': 0}
        self.blank_flag = 0
        self.space_flag = False
        self.next_flag = True
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" "] * 10
        
        for i in ascii_uppercase:
            self.ct[i] = 0
            
        # UI Setup
        self.root = tk.Tk()
        self.root.title("Sign Language Dashboard")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1200x800")
        self.root.configure(bg="#2c3e50")

        self.panel = tk.Label(self.root, bg="#34495e")
        self.panel.place(x=50, y=50, width=500, height=500)

        self.panel2 = tk.Label(self.root, bg="#34495e")
        self.panel2.place(x=650, y=100, width=400, height=400)

        self.T = tk.Label(self.root, text="Sign Language Interpreter", font=("Verdana", 24, "bold"), bg="#2c3e50", fg="white")
        self.T.place(x=50, y=5)

        self.T1 = tk.Label(self.root, text="Character:", font=("Verdana", 18), bg="#2c3e50", fg="#ecf0f1")
        self.T1.place(x=50, y=580)
        self.panel3 = tk.Label(self.root, text="", font=("Verdana", 24, "bold"), bg="#2c3e50", fg="#f1c40f")
        self.panel3.place(x=200, y=575)

        self.T3 = tk.Label(self.root, text="Sentence:", font=("Verdana", 18), bg="#2c3e50", fg="#ecf0f1")
        self.T3.place(x=50, y=640)
        self.panel5 = tk.Label(self.root, text="", font=("Verdana", 18), bg="#2c3e50", fg="#2ecc71", wraplength=1000, justify="left")
        self.panel5.place(x=200, y=640)

        # Suggestion Buttons
        self.b1 = tk.Button(self.root, text="", font=("Verdana", 12), width=15, command=self.action1)
        self.b1.place(x=100, y=720)
        self.b2 = tk.Button(self.root, text="", font=("Verdana", 12), width=15, command=self.action2)
        self.b2.place(x=300, y=720)
        self.b3 = tk.Button(self.root, text="", font=("Verdana", 12), width=15, command=self.action3)
        self.b3.place(x=500, y=720)
        self.b4 = tk.Button(self.root, text="", font=("Verdana", 12), width=15, command=self.action4)
        self.b4.place(x=700, y=720)

        self.btn_speak = tk.Button(self.root, text="SPEAK", font=("Verdana", 12, "bold"), bg="#e67e22", fg="white", command=self.speak_fun)
        self.btn_speak.place(x=950, y=715, width=100, height=40)
        self.btn_clear = tk.Button(self.root, text="CLEAR", font=("Verdana", 12, "bold"), bg="#c0392b", fg="white", command=self.clear_fun)
        self.btn_clear.place(x=1060, y=715, width=100, height=40)

        self.str = ""
        self.word = ""
        self.current_symbol = ""
        self.word1, self.word2, self.word3, self.word4 = "", "", "", ""
        
        print("Starting Dashboard Loop...")
        self.video_loop()

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok: return
            frame = cv2.flip(frame, 1)
            h_orig, w_orig, _ = frame.shape
            
            hands = hd.findHands(frame)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image.resize((500, 500)))
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            if hands:
                hand = hands[0]
                bx, by, bw, bh = hand['bbox']
                
                crop = frame[max(0, by-offset):min(h_orig, by+bh+offset), max(0, bx-offset):min(w_orig, bx+bw+offset)]
                white = np.ones((400, 400, 3), np.uint8) * 255
                
                handz = hd2.findHands(crop)
                if handz:
                    self.pts = handz[0]['lmList']
                    os_x = ((400 - bw) // 2) - 15
                    os_y = ((400 - bh) // 2) - 15
                    
                    # Drawing
                    connections = [(0,1,2,3,4), (5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]
                    for path in connections:
                        for i in range(len(path)-1):
                            p1, p2 = self.pts[path[i]], self.pts[path[i+1]]
                            cv2.line(white, (p1[0]+os_x, p1[1]+os_y), (p2[0]+os_x, p2[1]+os_y), (0, 255, 0), 3)
                    
                    cv2.line(white, (self.pts[5][0]+os_x, self.pts[5][1]+os_y), (self.pts[9][0]+os_x, self.pts[9][1]+os_y), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[9][0]+os_x, self.pts[9][1]+os_y), (self.pts[13][0]+os_x, self.pts[13][1]+os_y), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[13][0]+os_x, self.pts[13][1]+os_y), (self.pts[17][0]+os_x, self.pts[17][1]+os_y), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0]+os_x, self.pts[0][1]+os_y), (self.pts[5][0]+os_x, self.pts[5][1]+os_y), (0, 255, 0), 3)
                    cv2.line(white, (self.pts[0][0]+os_x, self.pts[0][1]+os_y), (self.pts[17][0]+os_x, self.pts[17][1]+os_y), (0, 255, 0), 3)
                    for i in range(21): cv2.circle(white, (self.pts[i][0]+os_x, self.pts[i][1]+os_y), 2, (0, 0, 255), 1)

                    self.predict(white)
                    
                    self.current_image2 = Image.fromarray(cv2.cvtColor(white, cv2.COLOR_BGR2RGB))
                    imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
                    self.panel2.imgtk = imgtk2
                    self.panel2.config(image=imgtk2)

            self.panel3.config(text=self.current_symbol)
            self.panel5.config(text=self.str)
            self.b1.config(text=self.word1)
            self.b2.config(text=self.word2)
            self.b3.config(text=self.word3)
            self.b4.config(text=self.word4)
            
        except:
            traceback.print_exc()
        finally:
            self.root.after(1, self.video_loop)

    def predict(self, white):
        white_input = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white_input, verbose=0)[0], dtype='float32')
        ch1 = np.argmax(prob)
        prob[ch1] = 0
        ch2 = np.argmax(prob)
        pl = [ch1, ch2]
        
        # --- FULL ORIGINAL LOGIC ---
        if pl in [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5], [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6], [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]): ch1 = 0
        if pl in [[2, 2], [2, 1]]:
            if (self.pts[5][0] < self.pts[4][0]): ch1 = 0
        if pl in [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]: ch1 = 2
        if pl in [[6, 0], [6, 6], [6, 2]]:
            if self.distance(self.pts[8], self.pts[16]) < 52: ch1 = 2
        if pl in [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]: ch1 = 3
        if pl in [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]:
            if self.pts[4][0] > self.pts[0][0]: ch1 = 3
        if pl in [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]:
            if self.pts[2][1] + 15 < self.pts[16][1]: ch1 = 3
        if pl in [[6, 4], [6, 1], [6, 2]]:
            if self.distance(self.pts[4], self.pts[11]) > 55: ch1 = 4
        if pl in [[1, 4], [1, 6], [1, 1]]:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]): ch1 = 4
        if pl in [[3, 6], [3, 4]]:
            if (self.pts[4][0] < self.pts[0][0]): ch1 = 4
        if pl in [[2, 2], [2, 5], [2, 4]]:
            if (self.pts[1][0] < self.pts[12][0]): ch1 = 4
        if pl in [[3, 6], [3, 5], [3, 4]]:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] > self.pts[10][1]: ch1 = 5
        if pl in [[3, 2], [3, 1], [3, 6]]:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][1] + 17 > self.pts[20][1]: ch1 = 5
        if pl in [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]:
            if self.pts[4][0] > self.pts[0][0]: ch1 = 5
        if pl in [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]: ch1 = 5
        if pl in [[5, 7], [5, 2], [5, 6]]:
            if self.pts[3][0] < self.pts[0][0]: ch1 = 7
        if pl in [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]:
            if self.pts[6][1] < self.pts[8][1]: ch1 = 7
        if pl in [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]:
            if self.pts[18][1] > self.pts[20][1]: ch1 = 7
        if pl in [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]:
            if self.pts[5][0] > self.pts[16][0]: ch1 = 6
        if pl in [[7, 2]]:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]: ch1 = 6
        if pl in [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]:
            if self.distance(self.pts[8], self.pts[16]) > 50: ch1 = 6
        if pl in [[4, 6], [4, 2], [4, 1], [4, 4]]:
            if self.distance(self.pts[4], self.pts[11]) < 60: ch1 = 6
        if pl in [[1, 4], [1, 6], [1, 0], [1, 2]]:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0: ch1 = 6
        if pl in [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0], [6, 3], [6, 4], [7, 5], [7, 2]]:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]): ch1 = 1
        if pl in [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]): ch1 = 1
        if pl in [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]): ch1 = 1
        if pl in [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])): ch1 = 1
        if pl in [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]: ch1 = 1
        if pl in [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]:
            if not (self.pts[0][0] + 13 < self.pts[8][0] and self.pts[0][0] + 13 < self.pts[12][0] and self.pts[0][0] + 13 < self.pts[16][0] and self.pts[0][0] + 13 < self.pts[20][0]) and not (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.distance(self.pts[4], self.pts[11]) < 50: ch1 = 1
        if pl in [[5, 0], [5, 5], [0, 1]]:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]: ch1 = 1

        res = ""
        if ch1 == 0:
            res = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]: res = 'A'
            elif self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]: res = 'T'
            elif self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]: res = 'E'
            elif self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]: res = 'M'
            elif self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]: res = 'N'
        elif ch1 == 2: res = 'C' if self.distance(self.pts[12], self.pts[4]) > 42 else 'O'
        elif ch1 == 3: res = 'G' if self.distance(self.pts[8], self.pts[12]) > 72 else 'H'
        elif ch1 == 7: res = 'Y' if self.distance(self.pts[8], self.pts[4]) > 42 else 'J'
        elif ch1 == 4: res = 'L'
        elif ch1 == 6: res = 'X'
        elif ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                res = 'Z' if self.pts[8][1] < self.pts[5][1] else 'Q'
            else: res = 'P'
        elif ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]): res = 'B'
            elif (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]): res = 'D'
            elif (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]): res = 'F'
            elif (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]): res = 'I'
            elif (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][1]): res = 'W'
            elif (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]: res = 'K'
            elif ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]): res = 'U'
            elif ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]): res = 'V'
            elif (self.pts[8][0] > self.pts[12][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]): res = 'R'

        if res in ['E', 'Y', 'B']:
            if (self.pts[4][0] < self.pts[5][0]) and (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]): res = "next"
        if res in ['next', 'B', 'C', 'H', 'F', 'X']:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and (self.pts[4][1] < self.pts[8][1] and self.pts[4][1] < self.pts[12][1] and self.pts[4][1] < self.pts[16][1] and self.pts[4][1] < self.pts[20][1]) and (self.pts[4][1] < self.pts[6][1] and self.pts[4][1] < self.pts[10][1] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]): res = 'Backspace'

        if res == "next" and self.prev_char != "next":
            target = self.ten_prev_char[(self.count-2)%10]
            if target == "Backspace": self.str = self.str[:-1]
            elif target != "next" and target != "Backspace" and target != " ": self.str += target
        
        self.prev_char = res
        self.current_symbol = res
        self.count += 1
        self.ten_prev_char[self.count%10] = res
        
        # Spelling Suggestions
        if HAS_ENCHANT and len(self.str.strip()) != 0:
            st = self.str.rfind(" "); ed = len(self.str)
            word = self.str[st+1:ed]
            self.word = word
            if len(word.strip()) != 0:
                sugs = ddd.suggest(word)
                self.word1 = sugs[0] if len(sugs) > 0 else ""
                self.word2 = sugs[1] if len(sugs) > 1 else ""
                self.word3 = sugs[2] if len(sugs) > 2 else ""
                self.word4 = sugs[3] if len(sugs) > 3 else ""
            else: self.word1 = self.word2 = self.word3 = self.word4 = ""

    def distance(self, x, y): return math.sqrt(((x[0]-y[0])**2) + ((x[1]-y[1])**2))
    def action1(self): self.apply_sug(self.word1)
    def action2(self): self.apply_sug(self.word2)
    def action3(self): self.apply_sug(self.word3)
    def action4(self): self.apply_sug(self.word4)
    
    def apply_sug(self, sug):
        if not sug: return
        idx_space = self.str.rfind(" ")
        self.str = self.str[:idx_space+1] + sug.upper()
        self.word1 = self.word2 = self.word3 = self.word4 = ""

    def speak_fun(self):
        self.speak_engine.say(self.str)
        self.speak_engine.runAndWait()

    def clear_fun(self):
        self.str = ""; self.word1 = self.word2 = self.word3 = self.word4 = ""

    def destructor(self):
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    Application().root.mainloop()
