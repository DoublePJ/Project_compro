import numpy as np
import cv2

CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
	"BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
	"DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
	"SOFA", "TRAIN", "TVMONITOR"]

from linebot import LineBotApi
from linebot.models import TextSendMessage

channel_access_token = "XRWXaniWWyBe1t8ZuSGm9ivz/x4DW8ov6OaI7x2E6SL2+LSpLP8p5pIb7RysfCXHXinaTnkdkwffdjTgtE6fijHgtKplktFtayeByiKFBL8GTsJ//U9h7Wekm3iBSZW2bA130mTGMR95eDB8GQzIVgdB04t89/1O/w1cDnyilFU="
line_bot_api = LineBotApi(channel_access_token)



colorframe = np.random.uniform(0,100, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD.prototxt","./MobileNetSSD/MobileNetSSD.caffemodel")

cap = cv2.VideoCapture("Cat.mp4")

while True:
    ret, frame = cap.read()
    
    if ret:
        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0,detections.shape[2]):
            percent = detections[0,0,i,2]
            if percent > 0.5:
                class_index = int(detections[0,0,i,1])
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype('int')

                label = "{} [{:.2f}%]".format(CLASSES[class_index], percent*100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), colorframe[class_index], 2)
                cv2.rectangle(frame, (startX-1, startY-30), (endX+1, startY), colorframe[class_index], cv2.FILLED)
                y = startY - 15 if startY-15>15 else startY+15
                cv2.putText(frame, label, (startX+20, y+5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
                
                if percent > 0.7:
                    print("find")
                    line_bot_api.push_message("U2dc21adc993dea4c1d229768022bab5f", TextSendMessage(text="COMPRO"))
                    
                
                else:
                    print("Non")
                
            

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xff==ord('q'):
            break


cap.release()
cv2.destroyAllWindows()

