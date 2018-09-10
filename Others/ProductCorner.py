from sklearn.cluster import KMeans, MeanShift, SpectralClustering
import shutil
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import csv
import json

with open("G:/浙大实习/数据label/JsonDataForBanner.csv") as fr:
    rows = csv.reader(fr)
    fileName = []
    ImageShape = []
    cornerAbsolutionCollection = []
    path = "G:/浙大实习/text_product聚类/productCorner.txt"
    fobj = open(path, 'w')
    for row in rows:
        img = Image.open("G:/浙大实习/allImages-8337/" + row[0])
        shape = img.size
        ImageShape.append(shape)
        name = row[0]

        item = json.loads(row[1])

        if item['productsNum'] == 1:
            tProduct = item['products'][0]['y']
            bProduct = item['products'][0]['y'] + item['products'][0]['height']
            lProduct = item['products'][0]['x']
            rProduct = item['products'][0]['x'] + item['products'][0]['width']
            cornerProduct = (tProduct, bProduct, lProduct, rProduct)
            fobj.write(
                '\n' + str(tProduct) + ' ' + str(bProduct) + ' ' + str(lProduct) + ' ' + str(rProduct) + ' ' + name)
    fobj.close()
