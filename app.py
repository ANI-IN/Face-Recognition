import cv2
import os

current_dir = os.getcwd()
recognizer = cv2.face.LBPHFaceRecognizer_create()

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

Web_cam = cv2.VideoCapture(0)

model_path = os.path.join(current_dir, 'trainer/trainer.yml')
Face_cascade = cv2.CascadeClassifier(cascade_path)
recognizer.read(model_path)

id = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, Img_frame = Web_cam.read()
    gray = cv2.cvtColor(Img_frame, cv2.COLOR_BGR2GRAY)
    faces = Face_cascade.detectMultiScale(gray, 1.3, 7)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(Img_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, confidence = recognizer.predict(roi_gray)
        
        if confidence < 75:
            if id == 1:
                id = 'Animesh'
                cv2.putText(Img_frame, id, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                
            elif id == 2:
                id = 'Prof. (Dr.) Kamal Ghanshala'
                cv2.putText(Img_frame, id, (100, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                
            elif id == 3:
                id = 'Rs rawat'
                cv2.putText(Img_frame, id, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        else:
            id = 'Unknown'
            cv2.putText(Img_frame, id, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
               
    cv2.imshow('Face Detection', Img_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

Web_cam.release()
cv2.destroyAllWindows()
