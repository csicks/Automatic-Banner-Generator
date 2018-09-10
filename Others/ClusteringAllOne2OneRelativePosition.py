from sklearn.cluster import KMeans, MeanShift, SpectralClustering, AffinityPropagation
import shutil
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import csv
import json
import math


def GetMin(n, c, shape):
    a = min(abs(n[0] - c[0]), abs(n[0] - c[1]), abs(n[1] - c[0]), abs(n[1] - c[1]))
    b = min(abs(n[2] - c[2]), abs(n[2] - c[3]), abs(n[3] - c[2]), abs(n[3] - c[3]))
    return min(a / shape[0], b / shape[1])


def Overlap(n, c):
    if c[0] < n[0] < c[1] and c[2] < n[2] < c[3]:
        return 1
    elif c[0] < n[0] < c[1] and c[2] < n[3] < c[3]:
        return 1
    elif c[0] < n[1] < c[1] and c[2] < n[2] < c[3]:
        return 1
    elif c[0] < n[1] < c[1] and c[2] < n[3] < c[3]:
        return 1
    elif n[0] < c[0] < n[1] and n[2] < c[3] < n[3]:
        return 1
    elif n[0] < c[1] < n[1] and n[2] < c[2] < n[3]:
        return 1
    elif n[0] < c[1] < n[1] and n[2] < c[3] < n[3]:
        return 1
    elif n[0] < c[0] < n[1] and n[2] < c[2] < n[3]:
        return 1
    elif c[0] < n[0] < n[1] < c[1] and n[2] < c[2] < c[3] < n[3]:
        return 1
    elif n[0] < c[0] < c[1] < n[1] and c[2] < n[2] < n[3] < c[3]:
        return 1
    else:
        return 0


with open("G:/浙大实习/数据label/JsonDataForBanner.csv") as fr:
    rows = csv.reader(fr)
    ImageShape = []
    cornerAbsolutionCollection = []
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

            centersPercent = []
            fourCorner = []
            for each in item["texts"]:
                x = each['x']
                y = each['y']
                width = each['width']
                height = each['height']
                fourCorner.append((y, y + height, x, x + width))

            th = 1 / 40
            cluster = []
            if len(fourCorner) > 0:
                k = 0
                for each in fourCorner:
                    if k == 0:
                        cluster.append([each])
                    else:
                        isInd = 1
                        state = 0
                        DD = []
                        ov = []
                        for aa in cluster:
                            distance = []
                            for ii in aa:
                                if Overlap(each, ii) == 1:
                                    ov.append(aa)
                                    state = 1
                                    break
                                distance.append(GetMin(each, ii, shape))
                            if state == 1:
                                continue
                            minD = min(distance)
                            if minD < th:
                                isInd = 0
                                DD.append([minD, cluster.index(aa)])
                        if state == 0:
                            if isInd == 1:
                                cluster.append([each])
                            else:
                                i = 1
                                mIndex = 0
                                mValue = DD[0][0]
                                while i < len(DD):
                                    if DD[i][0] < mValue:
                                        mIndex = i
                                    i += 1
                                num = DD[mIndex][1]
                                In = cluster[num]
                                cluster.remove(In)
                                In.append(each)
                                cluster.append(In)
                        else:
                            if len(ov) == 1:
                                In = ov[0]
                                cluster.remove(In)
                                In.append(each)
                                cluster.append(In)
                            else:
                                re = []
                                for cc in ov:
                                    re += cc
                                    cluster.remove(cc)
                                re.append(each)
                                cluster.append(re)
                    k += 1

            collection = []
            for each in cluster:
                if len(each) == 1:
                    collection.append(each[0])
                else:
                    tCollection = []
                    bCollection = []
                    lCollection = []
                    rCollection = []
                    for mm in each:
                        tCollection.append(mm[0])
                        bCollection.append(mm[1])
                        lCollection.append(mm[2])
                        rCollection.append(mm[3])
                    tText = min(tCollection)
                    bText = max(bCollection)
                    lText = min(lCollection)
                    rText = max(rCollection)
                    cornerText = (tText, bText, lText, rText)
                    collection.append(cornerText)

            k = 0
            eh = []
            for each in collection:
                if k == 0:
                    eh.append(each)
                else:
                    state = 0
                    for ww in eh:
                        if Overlap(each, ww) == 1:
                            eh.remove(ww)
                            tText = min(each[0], ww[0])
                            bText = max(each[1], ww[1])
                            lText = min(each[2], ww[2])
                            rText = max(each[3], ww[3])
                            eh.append((tText, bText, lText, rText))
                            state = 1
                    if state == 0:
                        eh.append(each)
                k += 1

            if len(eh) == 1:
                cornerAbsolutionCollection.append([eh, cornerProduct, name, shape])

    tagPic = []
    cc = []
    for element in cornerAbsolutionCollection:
        a1 = (element[0][0][0] - element[1][0]) / element[3][1]
        b1 = (element[0][0][1] - element[1][1]) / element[3][1]
        c1 = (element[0][0][2] - element[1][2]) / element[3][0]
        d1 = (element[0][0][3] - element[1][3]) / element[3][0]
        cc.append([[a1, b1, c1, d1], element[2]])

    tPo = []
    bPo = []
    lPo = []
    rPo = []
    for each in cc:
        tPo.append(each[0][0])
        bPo.append(each[0][1])
        lPo.append(each[0][2])
        rPo.append(each[0][3])
    X = np.array([tPo, bPo, lPo, rPo])
    X = X.T
    # y_C = MeanShift().fit_predict(X)
    y_C = MeanShift(bandwidth=0.4).fit_predict(X)
    for index, label in enumerate(y_C):
        tagPic.append([cc[index][1], label])

    ss = []
    for each in tagPic:
        xx = int(each[0][0: len(each[0]) - 4])
        ss.append([xx, each[0], str(each[1])])
    ss.sort()

    path = "G:/浙大实习/text_product聚类/resultRelativePosition.txt"
    fobj = open(path, 'w')
    fobj.write(str(len(ss)))
    for each in ss:
        fobj.write('\n' + each[1] + ' x' + each[2])
    fobj.close()
