import cv2
import os

face_id = input('Enter your ID : ')
Webcam = cv2.VideoCapture(0)        #To enable webcam using cv2 library

# Detect object in video stream using Haarcascade Frontal Face or Trained data set
face_detector = cv2.CascadeClassifier('C://Users//ANI-IN//AppData//Local//Programs//Python//Python310//Lib//cv2//data//haarcascade_frontalface_default.xml')

count = 0       #count no of faces(Each frame) Initialize to 0
while (True):

    _, Img_frame = Webcam.read()                            #Reading the frame using read function if something goes wrong it will return false

    gray = cv2.cvtColor(Img_frame, cv2.COLOR_BGR2GRAY)      #Converting particular Image frame to grayscale & storing into gray variable

    faces = face_detector.detectMultiScale(gray, 1.3, 5)    #It will return x,y,w,h coordinates, List of faces rectangle ,scaleFactor=1.3 minNeighbors=5
                                                            
    for (x, y, w, h) in faces:                              # Loops for each faces or particular frame so that it acts as a video frame
        
        cv2.rectangle(Img_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)        #Crop the Image
        count += 1
        cv2.imwrite("C:/Users/ANI-IN/Desktop/Project/dataset/user." + face_id + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])  #save the image

        
        cv2.imshow('Data Collection', Img_frame)       #Display the video frame, with bounded rectangle on the person's face
    
    if cv2.waitKey(1) & 0xFF == ord('q'):              #press q to stop the loop
        break
    
    elif count >= 300:                                 #If picture count is grater than or equal to 300 it will automatically stop the loop
        print("Process Done")
        break

Webcam.release()        # Stop video
cv2.destroyAllWindows() # Close all started windows