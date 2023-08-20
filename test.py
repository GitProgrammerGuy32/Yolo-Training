from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture('construction.mp4')
model = YOLO('ppe.pt')   # Copy The trained best.pt from Google Colab and paste it in your venv
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
               'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

bgrColor = (0,255,0)
while True:
    suc,cam = cap.read()
    result = model(cam, stream=True)

    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

            w,h = x2-x1,y2-y1
            cvzone.cornerRect(cam,(x1,y1,w,h))
            conf = math.ceil((box.conf[0]*100)) / 100
            cls = int(box.cls[0])

            cvzone.putTextRect(cam,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1)
    cv2.imshow("YOLO Custom Training", cam)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()