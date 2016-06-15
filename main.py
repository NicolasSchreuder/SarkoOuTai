__author__ = 'Nicolas Schreuder, Sholom Schechtman, Pierre Foret'

# coding=utf-8

import cv2
import numpy as np
import time
import scipy as sc
from Functions import *
import matplotlib.pyplot as plt
import joblib
from skimage.feature import daisy
import os

# This code takes the link of a YouTube video as input, after downloading the videos and extracting the faces
# present in the video, it returns the total length of when Sarkozy is on-screen
# You need an internet connection to run the code

initial_path = os.getcwd()

# We set the path for the haar cascades files to detect the faces and for the folders which would contain faces
cascPath = os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier(cascPath)
eye_cascade = cv2.CascadeClassifier(os.path.join(os.getcwd(), 'haarcascade_eye.xml'))
time1 = time.time()

# We take a YouTube link as input
video_link = raw_input("Hello ! Please enter the link of the video you want to analyze (type 'example' for demo) : ")

# Example link if the user types 'examples
if video_link == 'example':
    video_link = 'https://www.youtube.com/watch?v=vNrMckRyOP8'

# We download the video with PyTube, video_path is the path where the video is saved,
# is False is video was not downloaded
video_path = download_video_yt(video_link, os.getcwd())

# If invalid link entered :
while not video_path:
    video_link = raw_input('Please enter a valid link : ')
    video_path = download_video_yt(video_link, os.getcwd())

print("Your video has been downloaded, I a now processing it.")

# We create the folder where we are going to save the faces
os.chdir(video_path)
path_visages=os.path.join(os.getcwd(), 'Visages/')
if 'Visages' not in os.listdir(os.getcwd()):
    os.mkdir(path_visages)

# We read the video with OpenCV
cap = cv2.VideoCapture('Video.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)  # gets the video fps

print('This step may take a bit of time, but don\'t worry I am working on your video :) ! ')


w_video, h_video = cap.get(3), cap.get(4)  # w_video and h_video are getting the video resolution
n_pix = float(h_video*w_video)  # number of pixels for a h_video*w_video resolution

# Initialization: we initialize the variables we are going to use while streaming the video
temp = np.zeros([h_video, w_video, 3], np.float32)
n_frame = 0
frame_max = n_frame
n_shot = 0
frame_n_plan = 1.
ten = 10.*np.ones([h_video, w_video])
n_faces = 0  # number of faces
n_f_faces = 0  # number of frames from where we count the faces
shots = {}  # gives a list for each shot which is like :
# [number of faces,{frame:(faces coordinates,frame (open_cv)},list of the abscissa]
abs_plan = []
changing_pixels = []
time_plan0 = 0  # beginning time of a shot
time_plan1 = 0  # ending time of a shot


# We stream the video in the while loop
while cap.grab():
    n_frame += 1
    time_plan1 += 1/float(fps)
    ret, frame = cap.retrieve()
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    shape = frame1.shape

    # We compute the percentage of changing pixels between two consecutive frames
    diff = np.absolute(frame1-temp)
    pix_diff = 255.*sc.greater_equal(diff, ten)
    n_pix_diff = float(np.count_nonzero(pix_diff))
    changing_pixels.append(n_pix_diff/n_pix)

    # If the percentage is higher than our threshold we say that we have a new shot
    if n_pix_diff/n_pix > 0.8:
        # Saving the images in a folder
        if n_f_faces != 0 and n_faces != 0:
            save_faces(path_visages, n_shot, shots, abs_plan, n_faces, int(time_plan0), int(time_plan1),
                       w_video)
        time_plan0 = time_plan1
        n_shot += 1
        shots = {}
        abs_plan = []
        n_faces = 0
        n_f_faces = 0

    temp = frame
    if n_frame % 12 == 0:  # every 12 frames the faces are saved
        n_f_faces += 1
        # Face detection
        faces = faceCascade.detectMultiScale(frame1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces = list(faces)
        faces.sort(key=lambda x: x[0])
        faces = magnitude_order(faces)

        faces = embedded_faces(faces)
        if n_faces < len(faces):
            n_faces = len(faces)
        for (x, y, w, h) in faces:
            abs_plan.append(x)
        shots[n_frame] = (faces, frame)
save_faces(path_visages, n_shot, shots, abs_plan, n_faces, int(time_plan0), int(time_plan1), w_video)


os.chdir(video_path)
length_video = get_len('Video.mp4')

# We plot the evolution of the percentage of changing pixels according to time
# You can uncomment this part to see the graph
times = [i/25. for i in range(int(length_video+1)*25)][:len(changing_pixels)]
plt.plot(times, changing_pixels)
plt.xlabel('Time')
plt.ylabel('Percentage of changing pixels')
plt.axhline(0.8, color='r')
plt.show()

print('Task executed, you can find the video and the extracted faces in the following folder : ' + video_path)
print('Execution time : '+str(time.time()-time1) + " seconds.")

# We load the SVM trained and the PCA we used to reduce the size of the training vectors
SVM_path = os.path.join(initial_path, 'SVMRecognition/ClassifierSarko')
PCA_path = os.path.join(initial_path, 'SVMRecognition/PCASarko')
clf = joblib.load(SVM_path)
pca = joblib.load(PCA_path)
path = os.path.join(video_path, 'Visages')
os.chdir(path)


threshold = 0.8
YES = []
NO = []

# We go through all the face folders we created and evaluate if Sarkozy is in the folder or not with our SVM
for folder in os.listdir(path):
    if folder != '.DS_Store':
        folder_path = os.path.join(path, folder)
        os.chdir(folder_path)
        folders2 = os.listdir(os.getcwd())
        for folder2 in folders2:
            if folder2 != '.DS_Store' and folder2[0:4] != 'Icon':
                os.chdir(os.path.join(folder_path, folder2))
                images = os.listdir(os.getcwd())
                if '.DS_Store' in images:
                    images.remove('.DS_Store')
                n = len(images)
                oui = 0
                non = 1
                for image in images:
                    if image[0] == 'f':
                        face = cv2.imread(image, 0)
                        descs = daisy(face, step=5)
                        descs = np.asarray(descs).reshape(-1)
                        descs.reshape(1, -1)
                        descs = pca.transform(descs)
                        predict = clf.predict(descs)
                        if predict[0] == 1:
                            oui += 1
                percent = float(oui)/n
                if percent >= threshold:
                    YES.append((str(folder)+ '/' + folder2, percent))
                else:
                    NO.append((str(folder)+ '/' + folder2, percent))

YES.sort(key=sort_tuple_folder)
NO.sort(key=sort_tuple_folder)

# We print the results
print('Folders containing Sarkozy with confidence scores for Sarkozy : ', YES)
print('Folders containing other persons with confidence scores for Sarkozy :', NO)
on_screen_time = 0

percentage = 0
for i in range(len(YES)):
    folder = YES[i][0]
    begin, end = begin_end(folder)
    on_screen_time += (end-begin)
    percentage = on_screen_time/length_video*100

if on_screen_time == 0:
    print('I am almost sure that Sarkozy is not in the video.')
    os.chdir(initial_path)
    jogging=cv2.imread('Jogging.jpg')
    cv2.imshow('Joke', jogging)
    cv2.waitKey(0)
else:
    print('Sarkozy is appearing approximately '+str(on_screen_time)+' seconds in this video. ''That is ' +
          str(int(percentage))+'% of the video !')
    os.chdir(initial_path)

