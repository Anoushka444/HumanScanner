import numpy as np
from PIL import Image
import cv2
import os
def trainclass(dset_dir):
    #os.path.join joins the dset directory to (all) the files present init and the path to all images is then stored in the os.listdir
    path = [os.path.join(dset_dir, f) for f in os.listdir(dset_dir)]#os.listdir lists all the files in the dset_dir
            #appending all images in a list
    faces=[]
    ids=[] #faces correspond to the id of a particular user

    for image in path:
        img = Image.open(image).convert("L") #L turns image into grayscale
        imageNp= np.array(img,'uint8')#converting img to numpyarray
        id=int(os.path.split(image)[1].split(".")[1])
        #splitting the User string from the user id string , [1] signifies the 1st string in the path


        faces.append(imageNp)#putting the images in numpy format into the faces list
        ids.append(id)
    #converting id list into numpy format
    ids=np.array(ids)
    #feeding the classifier with the faces and ids list
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.yml")

trainclass("dataset")