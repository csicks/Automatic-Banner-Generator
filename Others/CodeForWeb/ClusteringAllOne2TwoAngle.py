from sklearn.cluster import KMeans, MeanShift, SpectralClustering, AffinityPropagation
import shutil
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import csv
import json
import math
from itertools import cycle


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


def AngleTwo(path_In_CSV, path_In_AllPic, path_Out_Class_All, path_Out_Class_Enum, path_Out_Class_Final):
    C1 = 4
    C2 = 2
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

                vcount = 0
                th3 = 1 / 10
                iTwo = []
                for each in eh1:
                    if (each[1] - each[0]) > shape[1] * th3 or (each[3] - each[2]) > shape[0] * th3:
                        vcount += 1
                        iTwo.append(eh1.index(each))

                if vcount == 2:
                    if (eh1[iTwo[0]][1] - eh1[iTwo[0]][0]) * (eh1[iTwo[0]][3] - eh1[iTwo[0]][2]) < (
                                eh1[iTwo[1]][1] - eh1[iTwo[1]][0]) * (eh1[iTwo[1]][3] - eh1[iTwo[1]][2]):
                        temp = eh1[iTwo[0]]
                        eh1[iTwo[0]] = eh1[iTwo[1]]
                        eh1[iTwo[1]] = temp
                    cornerAbsolutionCollection.append([eh1[iTwo[0]], eh1[iTwo[1]], cornerProduct, name, shape])

        tagPic = []
        cc = []
        for element in cornerAbsolutionCollection:
            y1 = ((element[0][0] + element[0][1]) / 2) / element[4][1]
            y2 = ((element[2][0] + element[2][1]) / 2) / element[4][1]
            x1 = ((element[0][2] + element[0][3]) / 2) / element[4][0]
            x2 = ((element[2][2] + element[2][3]) / 2) / element[4][0]
            if not IsMiddle(element[0], element[1]) == 1:
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
                cc.append([tx, ty, element[3]])

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
        count1 = np.zeros(nLabel + 3)
        for element in cornerAbsolutionCollection:
            y1 = ((element[0][0] + element[0][1]) / 2) / element[4][1]
            y2 = ((element[2][0] + element[2][1]) / 2) / element[4][1]
            if IsMiddle(element[0], element[1]) == 1:
                if y2 - y1 > th:
                    tagPic.append([element[3], nLabel])
                elif y1 - y2 > th:
                    tagPic.append([element[3], nLabel + 1])
                else:
                    tagPic.append([element[3], nLabel + 2])
        for index, label in enumerate(y_C):
            tagPic.append([cc[index][2], label])

        for each in tagPic:
            count1[each[1]] += 1

        ss = []
        for each in tagPic:
            xx = int(each[0][0: len(each[0]) - 4])
            ss.append([xx, each[0], str(each[1])])
        ss.sort()

        angle1 = []
        for each in ss:
            angle1.append([each[1], each[2]])

        tagPic = []
        cc = []
        for element in cornerAbsolutionCollection:
            y1 = ((element[1][0] + element[1][1]) / 2) / element[4][1]
            y2 = ((element[2][0] + element[2][1]) / 2) / element[4][1]
            x1 = ((element[1][2] + element[1][3]) / 2) / element[4][0]
            x2 = ((element[2][2] + element[2][3]) / 2) / element[4][0]
            if not IsMiddle(element[0], element[1]) == 1:
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
                cc.append([tx, ty, element[3]])

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
        count2 = []
        k = 0
        while k < C1:
            count2.append(np.zeros(nLabel + 3).tolist())
            k += 1
        for element in cornerAbsolutionCollection:
            y1 = ((element[1][0] + element[1][1]) / 2) / element[4][1]
            y2 = ((element[2][0] + element[2][1]) / 2) / element[4][1]
            if IsMiddle(element[0], element[1]) == 1:
                if y2 - y1 > th:
                    tagPic.append([element[3], nLabel])
                elif y1 - y2 > th:
                    tagPic.append([element[3], nLabel + 1])
                else:
                    tagPic.append([element[3], nLabel + 2])
        for index, label in enumerate(y_C):
            tagPic.append([cc[index][2], label])

        ss = []
        for each in tagPic:
            xx = int(each[0][0: len(each[0]) - 4])
            ss.append([xx, each[0], str(each[1])])
        ss.sort()

        path1 = path_Out_Class_All
        newfile = open(path1, 'w')
        newfile.write(str(len(angle1)) + '\n')
        k = 0
        for each in angle1:
            nText = each[0] + ' x' + each[1] + ' y' + ss[k][2] + '\n'
            newfile.write(nText)
            k += 1
        newfile.close()

        index1 = []
        tC1 = 0
        count1 = count1.tolist()
        while tC1 < C1:
            indexTemp = count1.index(max(count1))
            index1.append(indexTemp)
            count1[indexTemp] = 0
            tC1 += 1

        fobj = open(path1)
        lines = fobj.readlines()
        k = 0
        count = 0
        for line in lines:
            if k == 0:
                count = int(line[:-1])
            elif k != 0:
                if k != count:
                    line = line[:-1]
                sg = line.split(' ')
                for each in np.linspace(0, C1 - 1, C1):
                    if sg[1] == 'x' + str(index1[int(each)]):
                        count2[int(each)][int(sg[2][1])] += 1
            k += 1
        fobj.close()

        index2 = []
        for each in count2:
            tC2 = 0
            indexGroupTemp = []
            while tC2 < C2:
                indexTemp = each.index(max(each))
                indexGroupTemp.append(indexTemp)
                each[indexTemp] = 0
                tC2 += 1
            index2.append(indexGroupTemp)

        path2 = path_Out_Class_Enum
        fobj = open(path2, 'w')
        k = 0
        for each in index1:
            for aa in index2[k]:
                fobj.write('x' + str(each) + 'y' + str(aa) + '\n')
            k += 1
        fobj.close()

        fobj = open(path1, 'r')
        output = []
        pathF = path_Out_Class_Final
        lines = fobj.readlines()
        newfile = open(pathF, 'w')
        k = 0
        count = 0
        for line in lines:
            if k != 0:
                if line[len(line) - 1] == '\n':
                    line = line[:-1]
                sg = line.split(' ')
                state = 0
                kk = 0
                for each in index1:
                    for aa in index2[kk]:
                        if sg[1] == 'x' + str(each) and sg[2] == 'y' + str(aa):
                            state = 1
                    kk += 1
                if state == 1:
                    count += 1
            k += 1
        fobj.close()
        newfile.write(str(count))
        k = 0
        for line in lines:
            if k != 0:
                if line[len(line) - 1] == '\n':
                    line = line[:-1]
                sg = line.split(' ')
                state = 0
                kk = 0
                for each in index1:
                    for aa in index2[kk]:
                        if sg[1] == 'x' + str(each) and sg[2] == 'y' + str(aa):
                            state = 1
                    kk += 1
                if state == 1:
                    output.append([sg[0], sg[1] + sg[2]])
                    newfile.write('\n' + sg[0] + ' ' + sg[1] + sg[2])
            k += 1
        newfile.close()
        return output


path_In_CSV = "G:/浙大实习/数据label/JsonDataForBanner.csv"
path_In_AllPic = "G:/浙大实习/allImages-8337/"
path_Out_Class_All = "G:/浙大实习/text_product聚类/resultOne2TwoAngleBoth.txt"
path_Out_Class_Enum = "G:/浙大实习/text_product聚类/resultOne2TwoAngleClass.txt"
path_Out_Class_Final = "G:/浙大实习/text_product聚类/resultOne2TwoAngleFinal.txt"

output = AngleTwo(path_In_CSV, path_In_AllPic, path_Out_Class_All, path_Out_Class_Enum, path_Out_Class_Final)
x = 1
