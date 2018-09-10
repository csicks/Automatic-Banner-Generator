from sklearn.cluster import KMeans, MeanShift, SpectralClustering, AffinityPropagation
import shutil
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import csv
import json


def GetMin(n, c, shape):
    a = min(abs(n[0] - c[0]), abs(n[0] - c[1]), abs(n[1] - c[0]), abs(n[1] - c[1]))
    b = min(abs(n[2] - c[2]), abs(n[2] - c[3]), abs(n[3] - c[2]), abs(n[3] - c[3]))
    return [a / shape[1], b / shape[0]]


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


def ReleventPosition(n, c):
    if c[0] < n[0] < c[1] or c[0] < n[1] < c[1] or n[0] < c[0] < n[1] or n[0] < c[1] < n[1]:
        return 1
    elif c[2] < n[2] < c[2] or c[2] < n[3] < c[3] or n[2] < c[2] < n[3] or n[2] < c[3] < n[3]:
        return 2
    else:
        return 0


with open("G:/浙大实习/数据label/JsonDataForBanner.csv") as fr:
    rows = csv.reader(fr)
    ImageShape = []
    for row in rows:
        img = Image.open("G:/浙大实习/allImages-8337/" + row[0])
        shape = img.size
        ImageShape.append(shape)
        name = row[0]

        item = json.loads(row[1])

        if item['productsNum'] == 1:

            centersPercent = []
            fourCorner = []
            for each in item["texts"]:
                x = each['x']
                y = each['y']
                width = each['width']
                height = each['height']
                fourCorner.append((y, y + height, x, x + width))

            th = 1 / 15
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
                                else:
                                    if ReleventPosition(each, ii) == 1:
                                        distance.append(GetMin(each, ii, shape)[1])
                                    elif ReleventPosition(each, ii) == 2:
                                        distance.append(GetMin(each, ii, shape)[0])
                                    else:
                                        distance.append((GetMin(each, ii, shape)[0] + GetMin(each, ii, shape)[1]) / 2)
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

            k = 0
            eh1 = []
            th = 1 / 15
            for each in eh:
                if k == 0:
                    eh1.append(each)
                else:
                    state = 0
                    for ww in eh1:
                        dd=0
                        if ReleventPosition(each, ww) == 1:
                            dd=GetMin(each, ww, shape)[1]
                        elif ReleventPosition(each, ww) == 2:
                            dd=GetMin(each, ww, shape)[0]
                        else:
                            dd=(GetMin(each, ww, shape)[0] + GetMin(each, ww, shape)[1]) / 2
                        if dd <= th:
                            eh1.remove(ww)
                            tText = min(each[0], ww[0])
                            bText = max(each[1], ww[1])
                            lText = min(each[2], ww[2])
                            rText = max(each[3], ww[3])
                            eh1.append((tText, bText, lText, rText))
                            state = 1
                    if state == 0:
                        eh1.append(each)
                k += 1

            path = "G:/浙大实习/text聚类/"
            fobj = open(path + name[0:len(row[0]) - 4] + ".txt", 'w')
            th3 = 1 / 10
            for each in eh1:
                if (each[1] - each[0]) > shape[1] * th3 or (each[3] - each[2]) > shape[0] * th3:
                    fobj.write('\n' + str(each[0]) + ' ' + str(each[1]) + ' ' + str(each[2]) + ' ' + str(each[3]))
            fobj.close()

