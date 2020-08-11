#!/home/yashprime1/anaconda3/bin/python3
#!pip3 install dnspython
#!pip3 install pymongo[srv]
#!pip3 install tensorfow
import os
import tensorflow as tf

from PIL import Image, ImageOps
import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
np.set_printoptions(suppress=True)
EMOTIONS_LIST = ['FRONT','BACK','RIGHT','LEFT','STOP','No Input']    

model = tf.keras.models.load_model('/home/yashprime1/Desktop/AllFiles/hand_gestures/keras_model.h5')
video = cv2.VideoCapture("https://192.168.43.1:8080/video")
i=0


from controller import Robot
from controller import Keyboard
from datetime import datetime
import pymongo
import threading
previousInsertedId=0
count=0
from bson.objectid import ObjectId
def saveDirection(key):
  global count
  global previousInsertedId
  keyDirMaps={0:'Front',1:'Back',2:'Right',3:'Left',4:'Stop'}
  myClient = pymongo.MongoClient("mongodb+srv://test:test@cluster0.nkmaj.mongodb.net/directions?retryWrites=true&w=majority")
  myDb = myClient["directions"]
  myCol = myDb["directions"]
  myDir = { "direction": keyDirMaps[key]  , "implemented": count ,'time':datetime.now()}
  if count==0:
    x = myCol.insert_one(myDir)
    previousInsertedId=x.inserted_id
    print(keyDirMaps[key])
    print(previousInsertedId)
    
  else:
    x = myCol.replace_one({'_id':ObjectId(previousInsertedId)},myDir)
    print(keyDirMaps[key])
    print(previousInsertedId)
    
  count+=1
def retrieveData():
   myClient = pymongo.MongoClient("mongodb+srv://test:test@cluster0.nkmaj.mongodb.net/directions?retryWrites=true&w=majority")
   myDb = myClient["directions"]
   myCol = myDb["directions"]
   x=myCol.find()
   print(x[0])  

TIME_STEP = 64

robot = Robot()
timestep = int(robot.getBasicTimeStep())

keyboard=Keyboard()
keyboard.enable(timestep)

ds = []
dsNames = ['ds_right', 'ds_left']
for i in range(2):
    ds.append(robot.getDistanceSensor(dsNames[i]))
    ds[i].enable(TIME_STEP)
wheels = []
wheelsNames = ['wheel1', 'wheel2', 'wheel3', 'wheel4']
for i in range(4):
    wheels.append(robot.getMotor(wheelsNames[i]))
    wheels[i].setPosition(float('inf'))
    wheels[i].setVelocity(0.0)
while robot.step(TIME_STEP) != -1:
    _, fr =video.read()
    gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    print(fr.shape)
    print(gray_fr.shape)
    cv2.rectangle(fr, (100, 100), (800, 800), (0, 255, 0), 0)
    crop_image =gray_fr[100:800, 100:800]
    print(crop_image.shape)
    roi = cv2.resize(crop_image, (224, 224))
    cv2.imwrite(os.path.join('/home/yashprime1/Desktop/AllFiles/hand_gestures/train/capture', str(i)+'.jpg'),roi)
    roi=cv2.imread(os.path.join('/home/yashprime1/Desktop/AllFiles/hand_gestures/train/capture', str(i)+'.jpg')) 
    image_array=np.reshape(roi,(1,224,224,3))
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
    #normalized_image_array=np.reshape(normalized_image_array,(1,224,224,3))

    prediction = model.predict(normalized_image_array)
    k=np.argmax(prediction)
    print(prediction)
    cv2.putText(fr, EMOTIONS_LIST[k], (100, 100), font, 1, (255, 255, 0), 2)
    cv2.rectangle(fr,(100,100),(800,800),(255,0,0),2)
    cv2.imshow("hand",fr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #key=keyboard.getKey()
    #print(key)
    key=k
    print("Key :",EMOTIONS_LIST[key])
    leftSpeed = 0.0
    rightSpeed = 0.0
    if(key==0):
        th = threading.Thread(target=saveDirection,kwargs=dict(key=0))
        th.start()
        leftSpeed = 1
        rightSpeed = 1
    elif(key==1):
        th = threading.Thread(target=saveDirection,kwargs=dict(key=1))
        th.start()
        leftSpeed = -1
        rightSpeed = -1
    elif (key==2):
        th = threading.Thread(target=saveDirection,kwargs=dict(key=2))
        th.start()
        leftSpeed = 1
        rightSpeed = -1
    elif(key==3):
        th = threading.Thread(target=saveDirection,kwargs=dict(key=3))
        th.start()
        leftSpeed = -1
        rightSpeed = 1
    else:
        th = threading.Thread(target=saveDirection,kwargs=dict(key=4))
        th.start()
        leftSpeed = 0.0
        rightSpeed = 0.0
    wheels[0].setVelocity(leftSpeed)
    wheels[1].setVelocity(leftSpeed)
    wheels[2].setVelocity(rightSpeed)
    wheels[3].setVelocity(rightSpeed)
