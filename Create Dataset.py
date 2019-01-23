import cv2

#function to generate a dataset
def gen_dataset(img,id_user, img_id): #function to write images which will be used to train dataset
    cv2.imwrite("dataset/user."+str(id_user)+"."+str(img_id)+".jpg",img)



#defining func to draw boundary around feature(face,nose ,mouth)
def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text):
    gray_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #converts bgr to gray
    #multiscale func returns a list of features
    features = classifier.detectMultiScale(gray_img , scaleFactor ,minNeighbors) #detecting feature for classifier(mouth,nose etc)
    coordinate= [] #holds coordinates for face (x,y).(width, Height)
    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y),(x+w,y+h), color , 2) # 2 is thickness of border of rect
        cv2.putText(img,text, (x,y-4), cv2.FONT_HERSHEY_COMPLEX , 0.6 , color , 1 , cv2.LINE_AA)
        coordinate=[x,y,w,h]
    return coordinate

def detect(img, faceCascade,eyesCascade, noseCascade ,img_id):
    color={"blue":(255,0,0) , "red":(0,0,255), "green":(0,255,0) ,"white":(255,255,255)}
    coordinate  = draw_boundary(img,faceCascade, 1.1, 10 , color['blue'],"face")

    #detectMultiScale has coordinates for the face

    if len(coordinate)==4: #if coordinate isnt equaal to 4 face isnt detected (4 determines x,y,w,h)
        roi_img =img[coordinate[1]:coordinate[1]+coordinate[3], coordinate[0]:coordinate[0]+coordinate[2]]#crops using y:y+4 , x:x+4
        user_id=1
        gen_dataset(roi_img,user_id,img_id)

        #we need to detect the eyes from the newly cropped roi
        #coordinate = draw_boundary(roi_img, eyesCascade, 1.1, 14, color['red'], "eye")
        # detect the mouth from the newly cropped roi
        #coordinate = draw_boundary(roi_img, mouthCascade, 1.1, 20, color['white'], "mouth")
        # detect the nose from the newly cropped roi
        #coordinate = draw_boundary(roi_img, noseCascade, 1.1, 9, color['green'], "nose")

    return img

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
noseCascade = cv2.CascadeClassifier("Nose.xml")
mouthCascade = cv2.CascadeClassifier("Mouth.xml")





#creating object of video capture
#parameter is 0 for builtin webcam , -1 for external webcam
video_capture= cv2.VideoCapture(0)

img_id = 0




while True:
    _, img = video_capture.read()
    img = detect(img , faceCascade, eyesCascade, noseCascade ,img_id)
    #video_capture returns two parameter , we use only image so first parameter is underscore
    cv2.imshow("face detection",img)
    #imshow -- image show , first parameter is for the title of the window
    #second parameter is for the videosteam that takes image
    img_id+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #breaks loop if q pressed (terminates loop)
        break
video_capture.release()
cv2.destroyAllWindows()