from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import random
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

class MyGUI(QDialog):
    def __init__(self):
        super(MyGUI, self).__init__()
        loadUi("finalgui.ui", self)

        self.button.clicked.connect(self.buttonclicked)


        self.show()
        
        
    def buttonclicked(self):
        self.label_2.setText("")
        # words = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
        words = ["hack", "boy", "fly", "chunk", "funky"]
        curr_index = 0
        random_index = random.randint(0, len(words)-1)
        self.label.setText(words[random_index])
        
        
        word = words[random_index]
        self.label_3.setText(word[0])


        cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=1)
        classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

        offset = 20
        imgSize = 300

        folder = "data/Y"
        counter = 0

        labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

        while True:
            success, img = cap.read()
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            temp = 0
            # time.sleep(1)
            if hands:
                hand = hands[0]
                x,y,w,h = hand['bbox']

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

                crop = img[y-offset:y+h+offset,x-offset:x+w+offset]

                imgCropShape = crop.shape
                

                aspectRatio = h/w

                if aspectRatio>1:
                    k = imgSize/h
                    wCal = math.ceil(k*w)
                    if crop is None:
                        print('working')
                    else:
                        imgResize = cv2.resize(crop, (wCal, imgSize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgSize-wCal)/2)
                        imgWhite[:, wGap:wCal+wGap] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite)
                        print(prediction, index)
                    

                else:
                    k = imgSize/w
                    hCal = math.ceil(k*h)
                    if crop is None:
                        print('working')
                    else:
                        imgResize = cv2.resize(crop, (imgSize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgSize-hCal)/2)
                        imgWhite[hGap:hCal+hGap, :] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite)
                
                cv2.putText(imgOutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,128),2)
                temp = index

                # cv2.imshow("ImageCrop", crop)
                # cv2.imshow("ImageWhite", imgWhite)
            cv2.imshow("Image", imgOutput)
            cv2.waitKey(1)

            if((labels[temp]) == word[curr_index].upper() and curr_index == len(word) - 1):
                self.label_2.setText("Correct!")
                cv2.destroyAllWindows
                break     
            elif(labels[temp] == word[curr_index].upper()):
                curr_index += 1
                self.label_3.setText(word[curr_index])


    
    

        

def main():
    app = QApplication([])
    window = MyGUI()
    app.exec_()
    

if __name__ == '__main__':
    main()