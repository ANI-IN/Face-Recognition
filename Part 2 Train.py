import os
import cv2
import numpy as np                  #used for array computing
from PIL import Image;

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier('C://Users//ANI-IN//AppData//Local//Programs//Python//Python310//Lib//cv2//data//haarcascade_frontalface_default.xml')
path='dataSet'

def get_Img_Id(path):       #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]

    faceSamples=[]    #create empth face list
    Ids=[]            #create empty ID list

   
    for imagePath in imagePaths:                       #now looping through all the image paths and loading the Ids and the images
        
        Img=Image.open(imagePath).convert('L')         #loading the image and converting it to gray scale, 8-bit pixels, black and white
        Image_array=np.array(Img,'uint8')              #Now we are converting the Image into numpy array
       
        Id=int(os.path.split(imagePath)[-1].split(".")[1])       #getting the Id from the image
        print(Id)        
        
        
        faces=detector.detectMultiScale(Image_array)     # extract the face from the training image sample
        
        for (x,y,w,h) in faces:                          #If a face is there then append that in the list as well as Id of it
            faceSamples.append(Image_array[y:y+h,x:x+w])
            Ids.append(Id)

    return faceSamples,Ids

faces,Ids = get_Img_Id('C:/Users/ANI-IN/Desktop/Project/dataset/')      #Using getImgId put all the face into faces list and id into Ids list
trained_data = recognizer.train(faces, np.array(Ids))                   #Train the data   
print("Trained Successfully")
recognizer.write('C:/Users/ANI-IN/Desktop/Project/trainer/trainer.yml')     #Save the file into Trained.yml for further face recognition    
                                                                            #yet another markup language