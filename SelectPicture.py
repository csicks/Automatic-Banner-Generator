import sys
import os, os.path, shutil
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import kde
from scipy import optimize


# path1 = "G:/浙大实习/text_product聚类/result_4214_.txt"
# path2 = "G:/浙大实习/text_product聚类/SelectedPicture.txt"
# f1 = open(path1, 'r')
# f2 = open(path2, 'w')
# k = 0
# for line in f1:
#     if k == 0:
#         f2.write(line[:-1] + '\n')
#     else:
#         if line[len(line) - 1] == '\n':
#             line = line[:-1]
#         string = line.split('_')[0]
#         f2.write(string + '.jpg\n')
#     k += 1
# f1.close()
# f2.close()

def FindData(x, collection):
    for each in collection:
        if x == each:
            return 1
    return 0


def prob(x, k, mu, sigma):
    index = -0.5 * pow((x - mu) / sigma, 2)
    numerator = k * pow(np.e, index)
    denominator = sigma * pow(2 * np.pi, 0.5)
    return numerator / denominator


type0 = 2

path1 = "G:/浙大实习/text_product聚类/result_4214_.txt"
f1 = open(path1, 'r')
k = 0
ratio = ''
nameGroup = []
for line in f1:
    if k == 0:
        ratio = line[:-1]
    else:
        if line[len(line) - 1] == '\n':
            line = line[:-1]
        string = line.split('_')[0]
        nameGroup.append(string + '.jpg')
    k += 1
f1.close()

SPGroup = []

with open("G:/浙大实习/数据label/JsonDataForBanner.csv") as fr:
    rows = csv.reader(fr)
    for row in rows:
        if FindData(row[0], nameGroup) == 1:
            item = json.loads(row[1])
            if item['productsNum'] == 1:
                SPGroup.append(row[0])

if type0 == 1:
    path3 = "G:/浙大实习/text_product聚类/resultOne2OneAngle.txt"
    f3 = open(path3, 'r')
    k = 0
    SPGroupF = []
    ClassGroup = []
    count = 0
    for line in f3:
        if k != 0:
            if line[len(line) - 1] == '\n':
                line = line[:-1]
            string = line.split(' ')[0]
            if FindData(string, SPGroup) == 1:
                if line.split(' ')[1] not in ClassGroup:
                    ClassGroup.append(line.split(' ')[1])
                SPGroupF.append([string, line.split(' ')[1]])
                count += 1
        k += 1
    f3.close()

    path2 = "G:/浙大实习/text_product聚类/SelectedPictureOne.txt"
    f2 = open(path2, 'w')
    f2.write(str(count) + ' ' + str(len(ClassGroup)))
    for each in SPGroupF:
        f2.write('\n' + each[0] + ' ' + each[1])
    f2.close()

elif type0 == 2:
    path3 = "G:/浙大实习/text_product聚类/resultOne2TwoAngleFinal.txt"
    f3 = open(path3, 'r')
    k = 0
    SPGroupF = []
    ClassGroup = []
    count = 0
    for line in f3:
        if k != 0:
            if line[len(line) - 1] == '\n':
                line = line[:-1]
            string = line.split(' ')[0]
            if FindData(string, SPGroup) == 1:
                if line.split(' ')[1] not in ClassGroup:
                    ClassGroup.append(line.split(' ')[1])
                SPGroupF.append([string, line.split(' ')[1]])
                count += 1
        k += 1
    f3.close()

    path2 = "G:/浙大实习/text_product聚类/SelectedPictureTwo.txt"
    f2 = open(path2, 'w')
    f2.write(str(count) + ' ' + str(len(ClassGroup)))
    for each in SPGroupF:
        f2.write('\n' + each[0] + ' ' + each[1])
    f2.close()
