from sklearn.cluster import KMeans, MeanShift, SpectralClustering, AffinityPropagation
import shutil
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import csv
import json
import shutil

with open("G:/浙大实习/数据label/JsonDataForBanner.csv") as fr:
    rows = csv.reader(fr)
    path = "G:/浙大实习/text框/"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    for row in rows:
        name = row[0]
        item = json.loads(row[1])
        if item['productsNum'] == 1:
            TextBox=[]
            for each in item["texts"]:
                x = each['x']
                y = each['y']
                width = each['width']
                height = each['height']
                TextBox.append((y, y + height, x, x + width))
            fobj = open(path + name[0:len(name) - 4] + ".txt", 'w')
            for each in TextBox:
                fobj.write('\n' + str(each[0]) + ' ' + str(each[1]) + ' ' + str(each[2]) + ' ' + str(each[3]))
            fobj.close()