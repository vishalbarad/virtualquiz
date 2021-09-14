import cv2
from cvzone.HandTrackingModule import HandDetector
import pandas as pd
import cvzone
import time

# create VideoCapture() object
cap = cv2.VideoCapture(0)

# Set width and height of the video frame
cap.set(3,1280)
cap.set(4,720)

# Create HandDetector() object with probability=0.8 to detect hands
detector = HandDetector(detectionCon=0.8)

# read question-answer.csv file
df = pd.read_csv("/Users/vishalbarad/PycharmProjects/opencv/virtual quiz/question-answer.csv")
userans = None
score = 0
def update(cursor,bboxs,QuNo):
    for x,bbox in enumerate(bboxs):
        x1,y1,x2,y2 = bbox
        if x1 < cursor[0] < x2 and y1 < cursor[1] < y2:
            global userans
            userans = x+1
            if df['ans'][QuNo]==userans:
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),cv2.FILLED)
                global score
                score +=1
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), cv2.FILLED)
            return score
        else:
            userans = None


QuNo = 0
QuTotal = df.shape[0]
# Read video frame by frame
while(cap.isOpened()):
    ret, img = cap.read()
    if ret==True:
        img = cv2.flip(img,1) #flip the image
        hands,img = detector.findHands(img,flipType=False) #Find the hands

        if QuNo<QuTotal:
            img,bbox = cvzone.putTextRect(img,df['questions'][QuNo],[100,100],1.5,2,offset=20,colorR=5,border=3)
            img,bbox1 = cvzone.putTextRect(img,df['mcq1'][QuNo],[100,200],1.3,2,offset=30)
            img,bbox2 = cvzone.putTextRect(img,df['mcq2'][QuNo],[100,300],1.3,2,offset=30)
            img,bbox3 = cvzone.putTextRect(img,df['mcq3'][QuNo],[100,400],1.3,2,offset=30)
            img,bbox4 = cvzone.putTextRect(img,df['mcq4'][QuNo],[100,500],1.3,2,offset=30)

            if hands:
                lmList = hands[0]['lmList'] #hands[0]=> Right hand and hands[1]=>Left hand
                cursor = lmList[8]
                length, info = detector.findDistance(lmList[8],lmList[4])
                if length<60:
                    result = update(cursor,[bbox1,bbox2,bbox3,bbox4],QuNo)
                    if userans is not None:
                        time.sleep(0.3)
                        QuNo +=1
        else:
            score_ = round(score*100//QuTotal,2)
            img, _ = cvzone.putTextRect(img, "Quize completed", [250, 300], 2, 2, offset=30, border=5)
            img, _ = cvzone.putTextRect(img, f'Your score is {score_}%', [700, 300], 2, 2, offset=30, border=5)
            img, bbox = cvzone.putTextRect(img, "Retake quize", [700, 500],1.3,2,offset=30)
            lmList = hands[0]['lmList']  # hands[0]=> Right hand and hands[1]=>Left hand
            cursor = lmList[8]
            length, info = detector.findDistance(lmList[8], lmList[4])
            x1, y1, x2, y2 = bbox
            if (x1 < cursor[0] < x2 and y1 < cursor[1] < y2) and (length<60):
                cv2.rectangle(img, (x1, y1), (x2, y2), (225, 0, 0), cv2.FILLED)
                QuNo = 0
                score = 0
                userans = None
            else:
                pass

        perc = f'{round(QuNo*100//QuTotal)}%'
        img, _ = cvzone.putTextRect(img,perc,[1130,635],2,2,offset=16)

        cv2.imshow("video",img)
        key = cv2.waitKey(1)
        if key==ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()