# Sarkozy Vector Machine 

#### Program to check if Nicolas Sarkozy appears in a YouTube video (and if so, how long) .


Authors : Sholom Schechtman, Nicolas Schreuder & Pierre Foret


The code can easily be adapted to answer the question : "Is x in this video and if so, how long ?" where x is a given person.
In order to do this you just have to change the training set for the SVM.


main.py is the main python script, it takes a YouTube link as input and returns Sarkozy's exposition time in the video.

Functions.py contains the functions we use in the main.py file.

haarcascade_frontalface_default.xml is where the haar cascade is stored.

SVMRecognition contains the scripts to create a database and train a SVM on this database, it also contains the saved SVM.
