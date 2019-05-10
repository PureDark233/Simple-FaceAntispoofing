import cv2
import os
import numpy as np
from skimage import feature
import json
import glob
import tqdm


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
    imgdirs = glob.glob(
        os.path.join(root, folder, '*/*.jpg'))
    for file in tqdm.tqdm(imgdirs):
        if file.endswith('.jpg'):
            img = cv2.imread(file)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            h, s, v = cv2.split(hsv)
            y, cb, cr = cv2.split(ycbcr)
            h_hist, s_hist, v_hist = get_lbp_hist(np.array(h)), get_lbp_hist(np.array(s)), get_lbp_hist(np.array(v))
            y_hist, cb_hist, cr_hist = get_lbp_hist(np.array(y)), get_lbp_hist(np.array(cb)), get_lbp_hist(np.array(cr))
            totalhist = np.array([h_hist[0], s_hist[0], v_hist[0], y_hist[0], cb_hist[0], cr_hist[0]])
            totalhist = totalhist.flatten().tolist()
            totalhist.append(folder)
            data.append(totalhist)
    return data


def getmulti():
    totaldata = []
    for str in ('train_all/real', 'train_all/attack', 'test_all/real', 'test_all/attack'):
        # str = input("Enter the category, input 0 to stop\n")
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
        print('Done')


getmulti()
