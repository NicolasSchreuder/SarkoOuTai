__author__ = 'Nicolas Schreuder, Sholom Schechtman, Pierre Foret'

import subprocess, json
from pytube import YouTube
import re, os,cv2
import unicodedata


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nfkd_form.encode('ASCII', 'ignore')
    return only_ascii


def sort_tuple_folder(tuple):
    folder = tuple[0]
    folder=folder.split('_')
    return int(folder[0])


def magnitude_order(l):  # to take only same size faces
    if l != []:
        area = [l[i][2]*l[i][3] for i in range(len(l))]
        L2 = []
        M = max(area)
        for i in range(len(l)):
            if area[i] >= (3/4)*M:
                L2.append(l[i])
        return L2
    else:
        return []


def embedded_faces(l):
    """
    Checks if some of the faces are not from the same person on the same image
    """
    if l!=[]:
        l2 = [l[0]]
        x = l[0][0]
        w = l[0][3]
        y = l[0][1]
        h = l[0][2]

        for i in range(1, len(l)-1):
            x_pr = l[i][0]
            w_pr = l[i][3]
            y_pr = l[0][1]
            h_pr = l[0][2]
            k = True
            if x+w > x_pr:
                if y_pr+h_pr > y > y_pr:
                    if float((y_pr+h_pr-y)*(x+w-x_pr))/(w*h) > 0.5:
                        if w_pr * h_pr > w * h:
                            l2.pop()
                        else:
                            k = False
                elif y < y_pr and y_pr > y+h:
                    if float((y+h-y_pr)*(x+w-x_pr))/(w*h) > 0.5:
                        if w_pr * h_pr > w * h:
                            l2.pop()
                        else:
                            k = False

            if k:
                l2.append(l[i])
            x, w, y, h = x_pr, w_pr, y_pr, h_pr
        return l2
    else:
        return []


def cut_list(liste, n):
    """
    Given a list of abscisses and a number of classes, cluster them by finding the biggest gap between groups
    """
    abscissa = []
    liste = sorted(liste)
    distances = [(liste[i+1]-liste[i],i) for i in range(len(liste)-1)]
    distances.sort(key=lambda x: x[0])
    for i in range(n-1):
        abscissa.append(liste[distances[-1][1]])
        distances.pop()
        abscissa.sort()
    return abscissa


def get_len(filename):
    result = subprocess.Popen(["ffprobe", filename, '-print_format', 'json', '-show_streams', '-loglevel', 'quiet'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(json.loads(result.stdout.read())['streams'][0]['duration'])


def download_video_yt(link, output_path):
    """
    :param link: a Youtube link
    :param output_path: path where the downloaded video is saved
    :return:
    """
    try:
        video_folder = output_path  # folder where videos are put
        yt = YouTube(link)  # get the video
        video = yt.get('mp4', '360p')  # select the video in mp4 format
        yt.filename=re.sub(' ', '-', yt.filename)
        new_path=os.path.join(output_path, remove_accents(yt.filename))

        if not os.path.exists(new_path):  # We use this condition to see if the video has already been treated
            os.makedirs(new_path)
            yt.set_filename('Video')
            video.download(new_path)  # downloads the video

        return new_path  # return the path of the folder where the video has been downloaded
    except:
        print('The link is not valid !')
        return False


def save_faces(path_visages1, n_shot, shots1, abs_plan1, n_faces1, time_0, time_1, w_video):
    """ sorts and saves faces in folders by their position using the results of cut_list
     path_visages1 is a path folder, n_shot is the shot number,shots the dictionary
     defined in the main
     abs_plan a list of abscissa used to cluster, n_faces number of faces in a shot
     time_0 and time_1 the beginning and the end time of a shot (in seconds),
     w_video the width resolution of the video """

    abs = cut_list(abs_plan1, n_faces1)
    k = 0
    for frame_shot in shots1:
        k+=1
        visages_coord = shots1[frame_shot][0]  # coordinates of faces
        visages_frame = shots1[frame_shot][1]  # frame containing these faces
        if len(visages_coord) != n_faces1:
            # if there are less detected faces in this frame than we determined there are in a shot(the rare case)
            for vis in visages_coord:
                det_num_visage = False  # det_num_visage is True if we determined the cluster of this vis (=face)
                i = 0

                while not det_num_visage and i < len(abs):

                    if vis[0] < abs[i]:  # we are checking in which cluster (depending of i) should be the face
                        name_folder=path_visages1 + str(n_shot) + '_' + str(n_faces1)+'_t_' + \
                                    str(time_0)+'_' + str(time_1) + '/' + str(i)

                        if not os.path.exists(name_folder):
                            os.makedirs(name_folder)
                        os.chdir(name_folder)
                        x, y, w, h = vis[0], vis[1], vis[2], vis[3]
                        name = 'face'+str(k)+'.jpg'
                        up = y
                        bottom = y + h
                        left = x
                        right = min(x + h, w_video - 1)
                        face = cv2.resize(visages_frame[up:bottom, left:right], (64, 64), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(name, face)
                        det_num_visage = True
                    else:
                        i += 1

                if not det_num_visage:  # it means that we still did not determine the  face cluster (det_num_visage=False and i=len(abs)
                    # which means that face belongs to cluster the most on the right
                    name_folder=path_visages1 + str(n_shot) + '_' + str(n_faces1) + '_t_' + str(time_0)+'_'+str(time_1)+'/' + str(i)

                    if not os.path.exists(name_folder):
                        os.makedirs(name_folder)
                    os.chdir(name_folder)
                    x, y, w, h = vis[0], vis[1], vis[2], vis[3]
                    name = 'face' + str(k) + '.jpg'
                    up = y
                    bottom = y + h
                    left = x
                    right = min(x + h, w_video - 1)
                    face = cv2.resize(visages_frame[up:bottom, left:right], (64, 64),
                                        interpolation=cv2.INTER_AREA)
                    cv2.imwrite(name, face)
                    det_num_visage = True

        elif len(visages_coord) == n_faces1:

            # if the number of detected faces and the number of faces which
            # should be on the frame is the same the most frequent case

            for i in range(n_faces1):
                vis = visages_coord[i]
                name_folder = path_visages1 + str(n_shot) + '_' + str(n_faces1)+'_t_'+str(time_0)+'_'+str(time_1) + \
                              '/' + str(i)
                if not os.path.exists(name_folder):
                    os.makedirs(name_folder)
                os.chdir(name_folder)
                x, y, w, h = vis[0], vis[1], vis[2], vis[3]
                name = 'face' + str(k) + '.jpg'
                up = y
                bottom = y + h
                left = x
                right = min(x + h, w_video - 1)
                face = cv2.resize(visages_frame[up:bottom, left:right], (64, 64), interpolation=cv2.INTER_AREA)
                cv2.imwrite(name, face)


def begin_end(folder):
    """
    determines given the name of the folder, how much the person in the folder have appeared
    """
    k = 1
    while folder[-k] != '/':
        k += 1
    i = k+1
    while folder[-i] != '_':
        i += 1
    end = int(folder[-i+1:-k])
    j = i+1
    while folder[-j] != '_':
        j += 1
    beg = int(folder[-j+1:-i])
    return beg, end
