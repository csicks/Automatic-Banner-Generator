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


def IsMiddle(n, c):
    if n[2] < c[2] < c[3] < n[3] or c[2] < n[2] < n[3] < c[3]:
        return 1
    else:
        return 0


def AngleOne(path_In_CSV, path_In_AllPic, path_Out_Class):
    with open(path_In_CSV) as fr:
        rows = csv.reader(fr)
        ImageShape = []
        cornerAbsolutionCollection = []
        for row in rows:
            img = Image.open(path_In_AllPic + row[0])
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
                                            distance.append(
                                                (GetMin(each, ii, shape)[0] + GetMin(each, ii, shape)[1]) / 2)
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
                            dd = 0
                            if ReleventPosition(each, ww) == 1:
                                dd = GetMin(each, ww, shape)[1]
                            elif ReleventPosition(each, ww) == 2:
                                dd = GetMin(each, ww, shape)[0]
                            else:
                                dd = (GetMin(each, ww, shape)[0] + GetMin(each, ww, shape)[1]) / 2
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

                th3 = 1 / 10
                if len(eh1) == 1 and (
                                (eh1[0][1] - eh1[0][0]) > shape[1] * th3 or (eh1[0][3] - eh1[0][2]) > shape[0] * th3):
                    cornerAbsolutionCollection.append([eh1, cornerProduct, name, shape])

        tagPic = []
        cc = []
        for element in cornerAbsolutionCollection:
            y1 = ((element[0][0][0] + element[0][0][1]) / 2) / element[3][1]
            y2 = ((element[1][0] + element[1][1]) / 2) / element[3][1]
            x1 = ((element[0][0][2] + element[0][0][3]) / 2) / element[3][0]
            x2 = ((element[1][2] + element[1][3]) / 2) / element[3][0]
            if not IsMiddle(element[0][0], element[1]) == 1:
                tt = 0
                if x1 == x2 and y1 >= y2:
                    tt = math.pi / 2
                elif x1 == x2 and y1 < y2:
                    tt = -math.pi / 2
                elif y1 <= y2 and x1 > x2:
                    tt = math.atan((y2 - y1) / (x1 - x2))
                elif y1 > y2 and x1 > x2:
                    tt = math.atan((y1 - y2) / (x2 - x1)) + 2 * math.pi
                elif y1 <= y2 and x1 < x2:
                    tt = math.atan((y2 - y1) / (x1 - x2)) + math.pi
                else:
                    tt = math.atan((y2 - y1) / (x1 - x2)) + math.pi
                tx = math.cos(tt)
                ty = math.sin(tt)
                cc.append([tx, ty, element[2]])

        cex = []
        cey = []
        for each in cc:
            cex.append(each[0])
            cey.append(each[1])
        X = np.array([cex, cey])
        X = X.T
        yc = MeanShift()
        yc.fit_predict(X)
        y_C = yc.labels_
        nLabel = y_C.max() + 1
        th = 1 / 10
        for element in cornerAbsolutionCollection:
            y1 = ((element[0][0][0] + element[0][0][1]) / 2) / element[3][1]
            y2 = ((element[1][0] + element[1][1]) / 2) / element[3][1]
            if IsMiddle(element[0][0], element[1]) == 1:
                if y2 - y1 > th:
                    tagPic.append([element[2], nLabel])
                elif y1 - y2 > th:
                    tagPic.append([element[2], nLabel + 1])
                else:
                    tagPic.append([element[2], nLabel + 2])
        for index, label in enumerate(y_C):
            tagPic.append([cc[index][2], label])

        ss = []
        for each in tagPic:
            xx = int(each[0][0: len(each[0]) - 4])
            ss.append([xx, each[0], str(each[1])])
        ss.sort()

        output=[]
        path = path_Out_Class
        fobj = open(path, 'w')
        fobj.write(str(len(ss)))
        for each in ss:
            fobj.write('\n' + each[1] + ' x' + each[2])
            output.append([each[1],each[2]])
        fobj.close()
        return output


path_In_CSV = "G:/浙大实习/数据label/JsonDataForBanner.csv"
path_In_AllPic = "G:/浙大实习/allImages-8337/"
path_Out_Class = "G:/浙大实习/text_product聚类/resultOne2OneAngle.txt"

output=AngleOne(path_In_CSV,path_In_AllPic,path_Out_Class)
x=1