import cv2
import numpy as np
import glob,random,os
import time
from random import randrange
from enum import Enum
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:

    files = glob.glob("*.jpeg")
    file = random.choice(files)
    files2 = glob.glob("*.jpeg")
    file2 = random.choice(files2)
    files3 = glob.glob("*.jpeg")
    file3 = random.choice(files3)
    files4 = glob.glob("*.jpeg")
    file4 = random.choice(files4)
    files5 = glob.glob("*.jpeg")
    file5 = random.choice(files5)
    files6 = glob.glob("*.jpeg")
    file6 = random.choice(files6)
    files7 = glob.glob("*.jpeg")
    file7 = random.choice(files7)
    files8 = glob.glob("*.jpeg")
    file8 = random.choice(files8)
    files9 = glob.glob("*.jpeg")
    file9 = random.choice(files9)
    files10 = glob.glob("*.jpeg")
    file10 = random.choice(files10)
    files11 = glob.glob("*.jpeg")
    file11 = random.choice(files11)
    files12 = glob.glob("*.jpeg")
    file12 = random.choice(files12)
    files13 = glob.glob("*.jpeg")
    file13 = random.choice(files13)
    files14 = glob.glob("*.jpeg")
    file14 = random.choice(files14)
    files15 = glob.glob("*.jpeg")
    file15 = random.choice(files15)
    files16 = glob.glob("*.jpeg")
    file16 = random.choice(files16)
    files17 = glob.glob("*.jpeg")
    file17 = random.choice(files17)
    files18 = glob.glob("*.jpeg")
    file18 = random.choice(files18)
    files19 = glob.glob("*.jpeg")
    file19 = random.choice(files19)
    files20 = glob.glob("*.jpeg")
    file20 = random.choice(files20)
    files21 = glob.glob("*.jpeg")
    file21 = random.choice(files21)
    files22 = glob.glob("*.jpeg")
    file22 = random.choice(files22)
    files23 = glob.glob("*.jpeg")
    file23= random.choice(files23)
    files24 = glob.glob("*.jpeg")
    file24 = random.choice(files24)
    files25 = glob.glob("*.jpeg")
    file25 = random.choice(files25)
    files26 = glob.glob("*.jpeg")
    file26 = random.choice(files26)
    files27 = glob.glob("*.jpeg")
    file27 = random.choice(files27)
    files28 = glob.glob("*.jpeg")
    file28 = random.choice(files28)
    files29 = glob.glob("*.jpeg")
    file29= random.choice(files29)
    files30 = glob.glob("*.jpeg")
    file30 = random.choice(files30)
    files31 = glob.glob("*.jpeg")
    file31  = random.choice(files31)
    file32 = random.choice(files26)
    files32 = glob.glob("*.jpeg")
    file33 = random.choice(files27)
    files33 = glob.glob("*.jpeg")
    file34 = random.choice(files28)
    files34 = glob.glob("*.jpeg")
    file35= random.choice(files29)
    files35 = glob.glob("*.jpeg")
    file36 = random.choice(files30)
    files36 = glob.glob("*.jpeg")
    img1 = cv2.rotate(cv2.imread(file), cv2.cv2.ROTATE_90_CLOCKWISE)
    
    # Read Second Image
    img2 = cv2.rotate(cv2.imread(file2), cv2.cv2.ROTATE_90_CLOCKWISE)
    img3 = cv2.rotate(cv2.imread(file3), cv2.cv2.ROTATE_90_CLOCKWISE)
    
    # Read Second Image
    img4 = cv2.rotate(cv2.imread(file4), cv2.cv2.ROTATE_90_CLOCKWISE)
    img5 = cv2.rotate(cv2.imread(file5), cv2.cv2.ROTATE_90_CLOCKWISE)
    img6 = cv2.rotate(cv2.imread(file6), cv2.cv2.ROTATE_90_CLOCKWISE)
    img7 = cv2.rotate(cv2.imread(file7), cv2.cv2.ROTATE_90_CLOCKWISE)
    img8 = cv2.rotate(cv2.imread(file8), cv2.cv2.ROTATE_90_CLOCKWISE)
    img9 = cv2.rotate(cv2.imread(file9), cv2.cv2.ROTATE_90_CLOCKWISE)
    img10 = cv2.rotate(cv2.imread(file10), cv2.cv2.ROTATE_90_CLOCKWISE)
    img11 = cv2.rotate(cv2.imread(file11), cv2.cv2.ROTATE_90_CLOCKWISE)
    img12 = cv2.rotate(cv2.imread(file12), cv2.cv2.ROTATE_90_CLOCKWISE)
    img13 = cv2.rotate(cv2.imread(file13), cv2.cv2.ROTATE_90_CLOCKWISE)
    img14 = cv2.rotate(cv2.imread(file14), cv2.cv2.ROTATE_90_CLOCKWISE)
    img15 = cv2.rotate(cv2.imread(file15), cv2.cv2.ROTATE_90_CLOCKWISE)
    img16 = cv2.rotate(cv2.imread(file16), cv2.cv2.ROTATE_90_CLOCKWISE)
    img17 = cv2.rotate(cv2.imread(file17), cv2.cv2.ROTATE_90_CLOCKWISE)
    img18 = cv2.rotate(cv2.imread(file18), cv2.cv2.ROTATE_90_CLOCKWISE)
    img19 = cv2.rotate(cv2.imread(file19), cv2.cv2.ROTATE_90_CLOCKWISE)
    img20 = cv2.rotate(cv2.imread(file20), cv2.cv2.ROTATE_90_CLOCKWISE)
    img21 = cv2.rotate(cv2.imread(file21), cv2.cv2.ROTATE_90_CLOCKWISE)
    img22 = cv2.rotate(cv2.imread(file22), cv2.cv2.ROTATE_90_CLOCKWISE)
    img23 = cv2.rotate(cv2.imread(file23), cv2.cv2.ROTATE_90_CLOCKWISE)
    img24 = cv2.rotate(cv2.imread(file24), cv2.cv2.ROTATE_90_CLOCKWISE)
    img25 = cv2.rotate(cv2.imread(file25), cv2.cv2.ROTATE_90_CLOCKWISE)
    img26 = cv2.rotate(cv2.imread(file26), cv2.cv2.ROTATE_90_CLOCKWISE)
    img27 = cv2.rotate(cv2.imread(file27), cv2.cv2.ROTATE_90_CLOCKWISE)
    img28 = cv2.rotate(cv2.imread(file28), cv2.cv2.ROTATE_90_CLOCKWISE)
    img29 = cv2.rotate(cv2.imread(file29), cv2.cv2.ROTATE_90_CLOCKWISE) 
    img30 = cv2.rotate(cv2.imread(file30), cv2.cv2.ROTATE_90_CLOCKWISE)
    img31 = cv2.rotate(cv2.imread(file31), cv2.cv2.ROTATE_90_CLOCKWISE)
    img32 = cv2.rotate(cv2.imread(file32), cv2.cv2.ROTATE_90_CLOCKWISE)
    img33 = cv2.rotate(cv2.imread(file33), cv2.cv2.ROTATE_90_CLOCKWISE)
    img34 = cv2.rotate(cv2.imread(file34), cv2.cv2.ROTATE_90_CLOCKWISE)
    img35 = cv2.rotate(cv2.imread(file35), cv2.cv2.ROTATE_90_CLOCKWISE)
    img36 = cv2.rotate(cv2.imread(file36), cv2.cv2.ROTATE_90_CLOCKWISE)
    # concatenate image Horizontally
    Hori = np.concatenate((img1, img2, img3, img4, img5, img6, img7, img8, img9), axis=1)
    Hori2 = np.concatenate((img10, img11, img12, img13, img14, img15, img16, img17, img18), axis=1)
    Hori3 = np.concatenate((img19, img20, img21, img22, img23, img24, img25, img26, img27), axis=1)
    Hori4 = np.concatenate((img28, img29, img30, img31, img32, img33, img34, img35, img36), axis=1)
    Hori5 = np.concatenate((img20, img18, img19, img17, img2, img8, img20, img3, img8), axis=1)

    # concatenate image Vertically
    Verti = np.concatenate((Hori, Hori2, Hori3, Hori4, Hori5), axis=0)
    class text(Enum):
        connected = 1
        real  = 2
        i_exist = 3
        prepared = 4
        love = 5
        warmth = 6
        passion = 7
        drive = 8
        human = 9
        i_am_seen = 10
        hope = 11
        love_ = 12
        compassion = 13
        interconnectedness = 14
        cells = 15
        alive = 16
        free = 17
        strength = 18
        beach = 19
        #

    class HersheyFonts(Enum):
        cv2.FONT_HERSHEY_SIMPLEX        = 0,
        cv2.FONT_HERSHEY_PLAIN          = 1, 
        cv2.FONT_HERSHEY_DUPLEX         = 2, 
        cv2.FONT_HERSHEY_COMPLEX        = 3,
        cv2.FONT_HERSHEY_TRIPLEX        = 4, 
        cv2.FONT_HERSHEY_COMPLEX_SMALL  = 5, 
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX = 7
    
    cv2.putText(Verti, text(randrange(1,19)).name, (randrange(1920), randrange(1080)), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, randrange(2,10), (randrange(255), randrange(255), randrange(255)), randrange(5))
    cv2.putText(Verti, text(randrange(1,19)).name, (randrange(1920), randrange(1080)), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, randrange(10), (randrange(255), randrange(255), randrange(255)), randrange(5))
    cv2.putText(Verti, text(randrange(1,19)).name, (randrange(1920), randrange(1080)), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, randrange(10), (randrange(255), randrange(255), randrange(255)), randrange(5))
    cv2.putText(Verti, text(randrange(1,19)).name, (randrange(1920), randrange(1080)), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, randrange(10), (randrange(255), randrange(255), randrange(255)), randrange(5))
    cv2.putText(Verti, text(randrange(1,19)).name, (randrange(1920), randrange(1080)), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, randrange(10), (randrange(255), randrange(255), randrange(255)), randrange(5))
    cv2.putText(Verti, text(randrange(1,19)).name, (randrange(1920), randrange(1080)), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, randrange(10), (randrange(255), randrange(255), randrange(255)), randrange(5))

    height=1080
    width=1920
    blank_image = np.zeros((height,width,3), np.uint8)
    blank_image[:]=(0,0,255)


    #cv2.waitKey(randrange(20,340))
    _, frame = webcam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        textPos='blinking'

    elif gaze.is_right():
        textPos='right'
        for (x, y, w, h) in faces:
            cv2.putText(Verti, textPos, (x, y), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.6, (0, 255, 0), 2)
           # cv2.rectangle(Verti, ((randrange(0,500)),(randrange(255))), (x+10, y+10), ((randrange(255)), (randrange(255)), (randrange(255))), 2)


        cv2.imshow('connection', Verti)

    elif gaze.is_left():
        textPos='left'
        for (x, y, w, h) in faces:
            cv2.putText(Verti, textPos, (x, y), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.6, (0, 255, 0), 2)
            #cv2.rectangle(Verti, ((randrange(2500,4000)),(randrange(255))), (x+10, y+10), ((randrange(255)), (randrange(255)), (randrange(255))), 2)

        cv2.imshow('connection', Verti)

    elif gaze.is_center():
        textPos='center'
        for (x, y, w, h) in faces:
            cv2.putText(Verti, textPos, (x, y), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.6, (0, 255, 0), 2)
           # cv2.rectangle(Verti, ((randrange(4000)),(randrange(1000))), (x+10, y+10), ((randrange(255)), (randrange(255)), (randrange(255))), 2)

        cv2.waitKey(50)
        
        cv2.imshow('connection', Verti)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()



    if cv2.waitKey(1) == 27:
        break

