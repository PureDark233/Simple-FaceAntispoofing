import cv2
import os
import numpy as np
from skimage import feature
import json
import glob


def get_lbp_hist(channel):
    radius = 1
    n_point = radius * 8
    lbp = feature.local_binary_pattern(channel, n_point, radius, 'ror')
    max_bins = int(lbp.max() + 1)
    lbp_hist = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    return lbp_hist


def make_lbp_data(folder):
    root = os.getcwd()
    data = []
    t = 0
    imgdirs = glob.glob(
        os.path.join(root, folder, '*/*.jpg'))
    for file in imgdirs:
        t += 1
        if t % 100 != 0:
            continue
        if file.endswith('.jpg'):
            img = cv2.imread(file)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            hmat, smat, vmat = np.array(h) / 180, np.array(s) / 255, np.array(v) / 255
            h_hist, s_hist, v_hist = get_lbp_hist(hmat), get_lbp_hist(smat), get_lbp_hist(vmat)
            totalhist = []
            for i in h_hist[0]:
                totalhist.append(i)
            for i in s_hist[0]:
                totalhist.append(i)
            for i in v_hist[0]:
                totalhist.append(i)
            totalhist.append(folder)
            data.append(totalhist)
    return data


def getmulti():
    totaldata = []
    while 1:
        str = input("Enter the category, input 0 to stop\n")
        if str == '0':
            break
        else:
            if not os.path.isdir(str):
                print("No file in " + str)
            else:
                for i in make_lbp_data(str):
                    totaldata.append(i)
    with open('data.json', 'w') as f:
        json.dump(totaldata, f)


getmulti()
