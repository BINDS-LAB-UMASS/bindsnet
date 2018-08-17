import torch
import pickle
import os.path

# os.path.isfile(fname)


for i in range(1, 770601):
    if os.path.isfile("/home/devdhar/frames/frames/" + str(i) + ".frame"):
        print("file " + str(i))
        img = pickle.load(open("/home/devdhar/frames/frames/" + str(i) + ".frame", "rb"))
        for j in range(i+1, 770601):
            if os.path.isfile("/home/devdhar/frames/frames/"+str(j)+".frame"):
                check_img = pickle.load(open("/home/devdhar/frames/frames/"+str(j)+".frame", "rb"))
                if torch.equal(img, check_img):
                    os.remove("/home/devdhar/frames/frames/"+str(j)+".frame")
                    print("removed " + str(j))



