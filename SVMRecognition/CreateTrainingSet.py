__author__ = 'Nicolas Schreuder, Sholom Schechtman, Pierre Foret'
# -*- coding: utf-8 -*
from __future__ import print_function
import os
from PIL import Image


def processing(path):
    os.chdir(path)
    target_list1 = os.listdir(os.getcwd())
    target_list1.remove('.DS_Store')
    print("Liste des noms des visages du set: ", target_list1)
    for target_name1 in target_list1:
        os.chdir(target_name1)
        if not os.path.exists('processed_faces'):
            os.makedirs('processed_faces')
        os.chdir('faces')
        target_face_list1 = os.listdir(os.getcwd())
        target_face_list1.remove('.DS_Store')
        for target_face1 in target_face_list1:
            # IMAGE PROCESSING
            im = Image.open(target_face1)
            x_size, y_size = im.size
            if (x_size > x_reshape) and (y_size > y_reshape):
                out = im.resize((x_reshape, y_reshape))
                out.save(path+'/'+target_name1+'/processed_faces/'+target_face1, "JPEG")
        os.chdir(path)

if __name__ == '__main__':
    # Path containing the training faces
    dataset_folder_positive = os.path.join(os.getcwd(),'Sarko')
    dataset_folder_negative = os.path.join(os.getcwd(),'nonSarko')

    # Processing images

    x_reshape, y_reshape = 64, 64

    processing(dataset_folder_positive)
    processing(dataset_folder_negative)
