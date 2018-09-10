import numpy as np
import os, os.path, shutil
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import csv
import json
import copy
import random
import pickle

def TextBelong(x1, x2):
    if x1[0] >= x2[0] and x1[1] <= x2[1] and x1[2] >= x2[2] and x1[3] <= x2[3]:
        return 1
    else:
        return 0


def ReadDataTwo(numC, path_In_CSV, path_In_AllPic, path_In_Text, path_In_Source):
    path_S = path_In_Source
    fs = open(path_S, 'r').readlines()
    tp = fs[numC + 1][:-1].split(' ')
    hT1 = float(tp[2])
    wT1 = float(tp[3])
    hT2 = float(tp[6])
    wT2 = float(tp[7])
    TextBox = []
    with open(path_In_CSV) as fr:
        rows = csv.reader(fr)
        ImageShape = []
        for row in rows:
            img = Image.open(path_In_AllPic + row[0])
            shape = img.size
            ImageShape.append(shape)
            name = row[0]
            item = json.loads(row[1])
            if item['productsNum'] == 1:
                TextTemp = [name]
                for each in item['texts']:
                    x = each['x']
                    y = each['y']
                    width = each['width']
                    height = each['height']
                    line = each['lines']
                    TextTemp.append([y, y + height, x, x + width, line])
                TextBox.append(TextTemp)
    pathTextOut = path_In_Text
    collection1 = []
    collection2 = []
    dirTOut = os.listdir(pathTextOut)
    thT = 0.95
    for each in dirTOut:
        pic_name = each[:-4] + ".jpg"
        fOut = open(pathTextOut + each, 'r').readlines()
        if len(fOut) - 1 == 2:
            if fOut[1][len(fOut[1]) - 1] == '\n':
                fOut[1] = fOut[1][:-1]
            if fOut[2][len(fOut[2]) - 1] == '\n':
                fOut[2] = fOut[2][:-1]
            strO1 = fOut[1].split(' ')
            strO2 = fOut[2].split(' ')
            for i in range(0, len(strO1)):
                strO1[i] = int(strO1[i])
            for i in range(0, len(strO2)):
                strO2[i] = int(strO2[i])
            h1 = strO1[1] - strO1[0]
            w1 = strO1[3] - strO1[2]
            h2 = strO2[1] - strO2[0]
            w2 = strO2[3] - strO2[2]
            tempC = [pic_name]
            if thT < (h1 / w1) / (hT1 / wT1) < 1 / thT:
                tempA = []
                for aa in TextBox:
                    state = 0
                    if aa[0] == pic_name:
                        state = 1
                        k = 0
                        for bb in aa:
                            if k != 0:
                                if TextBelong(bb, strO1) == 1:
                                    tempA.append(
                                        [(bb[0] - strO1[0]) / (strO1[1] - strO1[0]),
                                         (bb[1] - strO1[0]) / (strO1[1] - strO1[0]),
                                         (bb[2] - strO1[2]) / (strO1[3] - strO1[2]),
                                         (bb[3] - strO1[2]) / (strO1[3] - strO1[2]),
                                         int(bb[4])])
                            k += 1
                    if state == 1:
                        break
                tempA.sort(key=lambda x: (x[1] - x[0]) / x[4], reverse=True)
                tempC.append(tempA)
                collection1.append(tempC)
            tempC = [pic_name]
            if thT < (h2 / w2) / (hT1 / wT1) < 1 / thT:
                tempA = []
                for aa in TextBox:
                    state = 0
                    if aa[0] == pic_name:
                        state = 1
                        k = 0
                        for bb in aa:
                            if k != 0:
                                if TextBelong(bb, strO2) == 1:
                                    tempA.append(
                                        [(bb[0] - strO2[0]) / (strO2[1] - strO2[0]),
                                         (bb[1] - strO2[0]) / (strO2[1] - strO2[0]),
                                         (bb[2] - strO2[2]) / (strO2[3] - strO2[2]),
                                         (bb[3] - strO2[2]) / (strO2[3] - strO2[2]),
                                         int(bb[4])])
                            k += 1
                    if state == 1:
                        break
                tempA.sort(key=lambda x: (x[1] - x[0]) / x[4], reverse=True)
                tempC.append(tempA)
                collection1.append(tempC)

            tempC = [pic_name]
            if thT < (h1 / w1) / (hT2 / wT2) < 1 / thT:
                tempA = []
                for aa in TextBox:
                    state = 0
                    if aa[0] == pic_name:
                        state = 1
                        k = 0
                        for bb in aa:
                            if k != 0:
                                if TextBelong(bb, strO1) == 1:
                                    tempA.append(
                                        [(bb[0] - strO1[0]) / (strO1[1] - strO1[0]),
                                         (bb[1] - strO1[0]) / (strO1[1] - strO1[0]),
                                         (bb[2] - strO1[2]) / (strO1[3] - strO1[2]),
                                         (bb[3] - strO1[2]) / (strO1[3] - strO1[2]),
                                         int(bb[4])])
                            k += 1
                    if state == 1:
                        break
                tempA.sort(key=lambda x: (x[1] - x[0]) / x[4], reverse=True)
                tempC.append(tempA)
                collection2.append(tempC)
            tempC = [pic_name]
            if thT < (h2 / w2) / (hT2 / wT2) < 1 / thT:
                tempA = []
                for aa in TextBox:
                    state = 0
                    if aa[0] == pic_name:
                        state = 1
                        k = 0
                        for bb in aa:
                            if k != 0:
                                if TextBelong(bb, strO2) == 1:
                                    tempA.append(
                                        [(bb[0] - strO2[0]) / (strO2[1] - strO2[0]),
                                         (bb[1] - strO2[0]) / (strO2[1] - strO2[0]),
                                         (bb[2] - strO2[2]) / (strO2[3] - strO2[2]),
                                         (bb[3] - strO2[2]) / (strO2[3] - strO2[2]),
                                         int(bb[4])])
                            k += 1
                    if state == 1:
                        break
                tempA.sort(key=lambda x: (x[1] - x[0]) / x[4], reverse=True)
                tempC.append(tempA)
                collection2.append(tempC)
    return [collection1, collection2]


def PointOverlap(group, x):
    for each in group:
        if each[0] < x < each[1]:
            return [1, each]
    return [0]


def LineOverlapLength(group, x):
    l = 0
    for each in group:
        if x[0] < each[0] < each[1] < x[1]:
            l += each[1] - each[0]
    return l


def LineOverlap(group, x):
    p1 = PointOverlap(group, x[0])
    p2 = PointOverlap(group, x[1])
    if p1[0] == 0:
        if p2[0] == 0:
            return LineOverlapLength(group, x) / (x[1] - x[0])
        elif p2[0] == 1:
            return (x[1] - p2[1][0] + LineOverlapLength(group, [x[0], p2[1][0]])) / (x[1] - x[0])
    elif p1[0] == 1:
        if p2[0] == 0:
            return (p1[1][1] - x[0] + LineOverlapLength(group, [p1[1][1], x[1]])) / (x[1] - x[0])
        elif p2[0] == 1:
            return (p1[1][1] - x[0] + x[1] - p2[1][0] + LineOverlapLength(group, [p1[1][1], p2[1][0]])) / (x[1] - x[0])


def OverlapAreaMask(r1, rP, mask):
    yP1 = rP[0]
    yP2 = rP[1]
    xP1 = rP[2]
    xP2 = rP[3]
    count = 0
    for i in range(int(r1[0]), int(r1[1])):
        for j in range(int(r1[2]), int(r1[3])):
            if yP1 < i < yP2 and xP1 < j < xP2 and mask[i - yP1][j - xP1] != 0:
                count += 1
    return count / ((r1[1] - r1[0]) * (r1[3] - r1[2]))


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


def LayoutTwo(path_In_CSV, path_In_AllPic, path_In_Text, path_In_SourceText, path_In_SourceProduct, path_Font,
              path_In_ChosenProduct, inputString, hP, wP,path_Out):
    def SortInput(x):
        temp = []
        k = 0
        for xx in x[1]:
            for each in xx:
                if each[0][4] == 1:
                    temp.append(0)
                else:
                    width = int((each[0][3] - each[0][2]) * wP)
                    height = int((each[0][1] - each[0][0]) / each[0][4] * hP)
                    n = int(width / height)
                    x1 = len(inputString[k]) % n
                    x2 = (len(inputString[k]) - x1) / n
                    if x1 != 0:
                        temp.append(each[0][4] - (x2 + 1))
                    else:
                        temp.append(each[0][4] - x2)
                k += 1
        return sum(temp)

    path_P = path_In_SourceProduct
    path_T = path_In_SourceText
    path_font = path_Font
    finalGroup = []
    for classNum in range(0, 8):
        fP = open(path_P, 'r').readlines()[classNum + 1]
        fT = open(path_T, 'r').readlines()[classNum + 1]
        yP1 = int((float(fP.split(' ')[0]) - 0.5 * float(fP.split(' ')[2])) * hP)
        yP2 = int((float(fP.split(' ')[0]) + 0.5 * float(fP.split(' ')[2])) * hP)
        xP1 = int((float(fP.split(' ')[1]) - 0.5 * float(fP.split(' ')[3])) * wP)
        xP2 = int((float(fP.split(' ')[1]) + 0.5 * float(fP.split(' ')[3])) * wP)
        yT11 = int((float(fP.split(' ')[0]) + float(fT.split(' ')[0]) - 0.5 * float(fT.split(' ')[2])) * hP)
        yT12 = int((float(fP.split(' ')[0]) + float(fT.split(' ')[0]) + 0.5 * float(fT.split(' ')[2])) * hP)
        xT11 = int((float(fP.split(' ')[1]) + float(fT.split(' ')[1]) - 0.5 * float(fT.split(' ')[3])) * wP)
        xT12 = int((float(fP.split(' ')[1]) + float(fT.split(' ')[1]) + 0.5 * float(fT.split(' ')[3])) * wP)
        yT21 = int((float(fP.split(' ')[0]) + float(fT.split(' ')[4]) - 0.5 * float(fT.split(' ')[6])) * hP)
        yT22 = int((float(fP.split(' ')[0]) + float(fT.split(' ')[4]) + 0.5 * float(fT.split(' ')[6])) * hP)
        xT21 = int((float(fP.split(' ')[1]) + float(fT.split(' ')[5]) - 0.5 * float(fT.split(' ')[7])) * wP)
        xT22 = int((float(fP.split(' ')[1]) + float(fT.split(' ')[5]) + 0.5 * float(fT.split(' ')[7])) * wP)
        TotalData1 = ReadDataTwo(classNum, path_In_CSV, path_In_AllPic, path_In_Text, path_In_SourceText)[0]
        TotalData2 = ReadDataTwo(classNum, path_In_CSV, path_In_AllPic, path_In_Text, path_In_SourceText)[1]

        th = 0.3
        TextGroup1 = []
        for data in TotalData1:
            pic_name = data[0]
            lenGroup = [[data[1][0][0], data[1][0][1]]]
            tGroup = [data[1][0]]
            for each in data[1][1:]:
                if LineOverlap(lenGroup, [each[0], each[1]]) < th:
                    lenGroup.append([each[0], each[1]])
                    tGroup.append(each)
            TextGroup1.append([pic_name, tGroup])
        TextGroup2 = []
        for data in TotalData2:
            pic_name = data[0]
            lenGroup = [[data[1][0][0], data[1][0][1]]]
            tGroup = [data[1][0]]
            for each in data[1][1:]:
                if LineOverlap(lenGroup, [each[0], each[1]]) < th:
                    lenGroup.append([each[0], each[1]])
                    tGroup.append(each)
            TextGroup2.append([pic_name, tGroup])

        num = len(inputString)
        TargetTextGroup = []
        for i in range(0, len(TextGroup1)):
            for j in range(0, len(TextGroup2)):
                if len(TextGroup1[i][1]) + len(TextGroup2[j][1]) == num:
                    TargetTextGroup.append(
                        [TextGroup1[i][0] + "_" + TextGroup2[j][0], [TextGroup1[i][1], TextGroup2[j][1]]])

        mapTBox = []
        for each in TargetTextGroup:
            mapTBox1 = []
            for i in range(0, len(each[1])):
                for j in range(0, len(each[1][i])):
                    if i == 0:
                        mapTBox1.append(
                            [(each[1][i][j][1] - each[1][i][j][0]) * (yT12 - yT11) / each[1][i][j][4], [i, j]])
                    if i == 1:
                        mapTBox1.append(
                            [(each[1][i][j][1] - each[1][i][j][0]) * (yT22 - yT21) / each[1][i][j][4], [i, j]])
            mapTBox1.sort(reverse=True)
            mapTBox.append(mapTBox1)

        NewTargetTextGroup = []
        for k in range(len(TargetTextGroup)):
            temp2 = []
            for i in range(0, len(TargetTextGroup[k][1])):
                temp3 = []
                for j in range(0, len(TargetTextGroup[k][1][i])):
                    if i == 0:
                        temp3.append([TargetTextGroup[k][1][i][j], mapTBox[k].index(
                            [(TargetTextGroup[k][1][i][j][1] - TargetTextGroup[k][1][i][j][0]) * (yT12 - yT11) /
                             TargetTextGroup[k][1][i][j][4],
                             [i, j]])])
                    if i == 1:
                        temp3.append([TargetTextGroup[k][1][i][j], mapTBox[k].index(
                            [(TargetTextGroup[k][1][i][j][1] - TargetTextGroup[k][1][i][j][0]) * (yT22 - yT21) /
                             TargetTextGroup[k][1][i][j][4],
                             [i, j]])])
                temp2.append(temp3)
            NewTargetTextGroup.append([TargetTextGroup[k][0], temp2])

        SuitableTextGroup = []
        for each in NewTargetTextGroup:
            k = 0
            state = 0
            while k < num:
                for i in range(0, len(each[1])):
                    for j in range(0, len(each[1][i])):
                        if each[1][i][j][1] == k:
                            if i == 0 and int((each[1][i][j][0][3] - each[1][i][j][0][2]) * (xT12 - xT11) / (
                                            (each[1][i][j][0][1] - each[1][i][j][0][0]) * (yT12 - yT11) /
                                        each[1][i][j][0][4])) * each[1][i][j][0][4] >= len(inputString[k]):
                                state += 1
                            if i == 1 and int((each[1][i][j][0][3] - each[1][i][j][0][2]) * (xT22 - xT21) / (
                                            (each[1][i][j][0][1] - each[1][i][j][0][0]) * (yT22 - yT21) /
                                        each[1][i][j][0][4])) * each[1][i][j][0][4] >= len(inputString[k]):
                                state += 1
                k += 1
            if state == num:
                SuitableTextGroup.append(each)
        SuitableTextGroup.sort(key=SortInput)

        th_d = 0.2
        th_h_r = 1.5
        pic_number = 0

        imp = Image.open(path_In_ChosenProduct)
        imp1 = imp.resize((int(imp.size[0] * (yP2 - yP1) / imp.size[1]), yP2 - yP1))
        xP1Temp = int((xP1 + xP2 - int(imp.size[0] * (yP2 - yP1) / imp.size[1])) / 2)
        xP2Temp = int((xP1 + xP2 + int(imp.size[0] * (yP2 - yP1) / imp.size[1])) / 2)
        xP1 = xP1Temp
        xP2 = xP2Temp
        r, g, b, a = imp1.split()
        mask = np.array(a)
        mask = mask.tolist()

        classGroup = [[yP1, yP2, xP1, xP2]]
        classGroupText = []
        for layout in SuitableTextGroup:
            cGroupTemp = []
            lTemp1 = copy.deepcopy(layout[1][0])
            lTemp2 = copy.deepcopy(layout[1][1])
            lTemp1.sort(key=lambda x: x[0][0])
            lTemp2.sort(key=lambda x: x[0][0])
            mapBox1 = []
            mapBox2 = []
            for each in lTemp1:
                mapBox1.append(layout[1][0].index(each))
            for each in lTemp2:
                mapBox2.append(layout[1][1].index(each))
            for i in range(0, len(lTemp1) - 1):
                if lTemp1[i][0][1] > lTemp1[i + 1][0][0] or lTemp1[i + 1][0][0] - lTemp1[i][0][1] < (
                            lTemp1[i + 1][0][1] - lTemp1[i + 1][0][0]) * th_d:
                    if (lTemp1[i][0][1] - lTemp1[i][0][0]) / (lTemp1[i + 1][0][1] - lTemp1[i + 1][0][0]) > th_h_r:
                        lTemp1[i][0][1] = lTemp1[i + 1][0][0] - (lTemp1[i + 1][0][1] - lTemp1[i + 1][0][0]) * th_d
                    elif (lTemp1[i + 1][0][1] - lTemp1[i + 1][0][0]) / (lTemp1[i][0][1] - lTemp1[i][0][0]) > th_h_r:
                        lTemp1[i + 1][0][0] = lTemp1[i][0][1] + (lTemp1[i][0][1] - lTemp1[i][0][0]) * th_d
                    else:
                        ratio = (lTemp1[i][0][1] - lTemp1[i][0][0]) / (
                            lTemp1[i + 1][0][1] - lTemp1[i + 1][0][0] + lTemp1[i][0][1] - lTemp1[i][0][0])
                        dd = 0
                        if (lTemp1[i + 1][0][1] - lTemp1[i + 1][0][0]) <= (lTemp1[i][0][1] - lTemp1[i][0][0]):
                            dd = (lTemp1[i + 1][0][1] - lTemp1[i + 1][0][0]) * th_d + lTemp1[i][0][1] - \
                                 lTemp1[i + 1][0][0]
                        else:
                            dd = (lTemp1[i][0][1] - lTemp1[i][0][0]) * th_d + lTemp1[i][0][1] - lTemp1[i + 1][0][0]
                        lTemp1[i][0][1] = lTemp1[i][0][1] - ratio * dd
                        lTemp1[i + 1][0][0] = lTemp1[i + 1][0][0] + (1 - ratio) * dd
            for i in range(0, len(lTemp1)):
                layout[1][0][mapBox1[i]] = lTemp1[i]
            for i in range(0, len(lTemp2) - 1):
                if lTemp2[i][0][1] > lTemp2[i + 1][0][0] or lTemp2[i + 1][0][0] - lTemp2[i][0][1] < (
                            lTemp2[i + 1][0][1] - lTemp2[i + 1][0][0]) * th_d:
                    if (lTemp2[i][0][1] - lTemp2[i][0][0]) / (lTemp2[i + 1][0][1] - lTemp2[i + 1][0][0]) > th_h_r:
                        lTemp2[i][0][1] = lTemp2[i + 1][0][0] - (lTemp2[i + 1][0][1] - lTemp2[i + 1][0][0]) * th_d
                    elif (lTemp2[i + 1][0][1] - lTemp2[i + 1][0][0]) / (lTemp2[i][0][1] - lTemp2[i][0][0]) > th_h_r:
                        lTemp2[i + 1][0][0] = lTemp2[i][0][1] + (lTemp2[i][0][1] - lTemp2[i][0][0]) * th_d
                    else:
                        ratio = (lTemp2[i][0][1] - lTemp2[i][0][0]) / (
                            lTemp2[i + 1][0][1] - lTemp2[i + 1][0][0] + lTemp2[i][0][1] - lTemp2[i][0][0])
                        dd = 0
                        if (lTemp2[i + 1][1] - lTemp2[i + 1][0][0]) <= (lTemp2[i][0][1] - lTemp2[i][0][0]):
                            dd = (lTemp2[i + 1][0][1] - lTemp2[i + 1][0][0]) * th_d + lTemp2[i][0][1] - \
                                 lTemp2[i + 1][0][0]
                        else:
                            dd = (lTemp2[i][0][1] - lTemp2[i][0][0]) * th_d + lTemp2[i][0][1] - lTemp2[i + 1][0][0]
                        lTemp2[i][0][1] = lTemp2[i][0][1] - ratio * dd
                        lTemp2[i + 1][0][0] = lTemp2[i + 1][0][0] + (1 - ratio) * dd
            for i in range(0, len(lTemp2)):
                layout[1][1][mapBox2[i]] = lTemp2[i]

            mapBoxTemp = []
            for i in range(len(layout[1])):
                for j in range(len(layout[1][i])):
                    if i == 0:
                        mapBoxTemp.append(
                            [(layout[1][i][j][0][1] - layout[1][i][j][0][0]) * (yT12 - yT11) / layout[1][i][j][0][4],
                             [i, j]])
                    if i == 1:
                        mapBoxTemp.append(
                            [(layout[1][i][j][0][1] - layout[1][i][j][0][0]) * (yT22 - yT21) / layout[1][i][j][0][4],
                             [i, j]])
            mapBoxTemp.sort(reverse=True)

            for i in range(0, len(lTemp1)):
                for j in range(0, len(layout[1][0])):
                    if lTemp1[i][0] == layout[1][0][j][0]:
                        lTemp1[i] = layout[1][0][j]

            for i in range(0, len(lTemp2)):
                for j in range(0, len(layout[1][1])):
                    if lTemp2[i][0] == layout[1][1][j][0]:
                        lTemp2[i] = layout[1][1][j]

            tagBox1 = []
            for i in range(0, len(lTemp1)):
                middleD = ((layout[1][0][i][0][2] + layout[1][0][i][0][3]) / 2 - (
                    layout[1][0][0][0][2] + layout[1][0][0][0][3]) / 2) / (
                              layout[1][0][i][0][3] - layout[1][0][i][0][2])
                if middleD <= -1 / 4:
                    tagBox1.append(-1)
                elif middleD >= 1 / 4:
                    tagBox1.append(1)
                else:
                    tagBox1.append(0)

            n11 = tagBox1.count(-1)
            n12 = tagBox1.count(0)
            n13 = tagBox1.count(1)
            tagBox2 = []
            for i in range(0, len(lTemp2)):
                middleD = ((layout[1][1][i][0][2] + layout[1][1][i][0][3]) / 2 - (
                    layout[1][1][0][0][2] + layout[1][1][0][0][3]) / 2) / (
                              layout[1][1][i][0][3] - layout[1][1][i][0][2])
                if middleD <= -1 / 4:
                    tagBox2.append(-1)
                elif middleD >= 1 / 4:
                    tagBox2.append(1)
                else:
                    tagBox2.append(0)

            n21 = tagBox2.count(-1)
            n22 = tagBox2.count(0)
            n23 = tagBox2.count(1)

            upGroup = []
            downGroup = []
            for i in range(0, len(lTemp1)):
                yy = yT11 + int(
                    layout[1][0][i][0][0] * (yT12 - yT11) - (layout[1][0][i][0][1] - layout[1][0][i][0][0]) * (
                        yT12 - yT11) * 0.2 + (layout[1][0][i][0][1] - layout[1][0][i][0][0]) * (yT12 - yT11) * 0.5)
                if yy < hP / 2:
                    upGroup.append(layout[1][0][i])
                else:
                    downGroup.append(layout[1][0][i])

            for i in range(0, len(lTemp2)):
                yy = yT21 + int(
                    layout[1][1][i][0][0] * (yT22 - yT21) - (layout[1][1][i][0][1] - layout[1][1][i][0][0]) * (
                        yT22 - yT21) * 0.2 + (layout[1][1][i][0][1] - layout[1][1][i][0][0]) * (yT22 - yT21) * 0.5)
                if yy < hP / 2:
                    upGroup.append(layout[1][1][i])
                else:
                    downGroup.append(layout[1][1][i])

            state_Pic = 0
            th_OA = 0.5
            while len(upGroup) != 0 or len(downGroup) != 0:
                for i in range(0, len(lTemp1)):
                    width = int((layout[1][0][i][0][3] - layout[1][0][i][0][2]) * (xT12 - xT11))
                    height = int(
                        (layout[1][0][i][0][1] - layout[1][0][i][0][0]) * (yT12 - yT11) / layout[1][0][i][0][4])
                    rr = random.randint(1, 3)
                    if layout[1][0][i][1] > 3:
                        font = ImageFont.truetype(path_font + str(3) + '/' + str(rr) + ".ttf", height, encoding="unic")
                    else:
                        font = ImageFont.truetype(path_font + str(layout[1][0][i][1]) + '/' + str(rr) + ".ttf", height,
                                                  encoding="unic")
                    if layout[1][0][i] in upGroup:
                        upGroup.remove(layout[1][0][i])
                        if layout[1][0][i][0][4] == 1:
                            widthW = font.getsize(inputString[layout[1][0][i][1]])[0]
                            if n12 >= n11 and n12 >= n13:
                                startP = 1 / 2 * (xT12 - xT11) - widthW / 2
                            elif n11 >= n12 and n11 >= n13:
                                startP = 0
                            elif n13 >= n11 and n13 >= n12:
                                startP = (xT12 - xT11) - widthW
                            y1 = yT11 + int(layout[1][0][i][0][0] * (yT12 - yT11))
                            y2 = yT11 + int(layout[1][0][i][0][0] * (yT12 - yT11)) + int(height)
                            x1 = xT11 + int(startP)
                            x2 = x1 + font.getsize(inputString[layout[1][0][i][1]][0])[0]
                            maskX1 = 0
                            maskX2 = 0
                            for ii in range(1, len(mask[int((y1 + y2) / 2 - yP1)])):
                                if mask[int((y1 + y2) / 2 - yP1)][ii] != 0 and mask[int((y1 + y2) / 2 - yP1)][
                                            ii - 1] == 0:
                                    maskX1 = xP1 + ii
                                if mask[int((y1 + y2) / 2 - yP1)][ii] != 0 and mask[int((y1 + y2) / 2 - yP1)][
                                            ii + 1] == 0:
                                    maskX2 = xP1 + ii
                                    break
                            wordGroup = [[y1, y2, x1, x2]]
                            for jj in range(1, len(inputString[layout[1][0][i][1]])):
                                x1 = x2
                                x2 += font.getsize(inputString[layout[1][0][i][1]][jj])[0]
                                wordGroup.append([y1, y2, x1, x2])
                            stateWord = 0
                            for jj in range(0, len(inputString[layout[1][0][i][1]])):
                                OA = OverlapAreaMask(
                                    (wordGroup[jj][0], wordGroup[jj][1], wordGroup[jj][2], wordGroup[jj][3]),
                                    (yP1, yP2, xP1, xP2), mask)
                                if Overlap((wordGroup[jj][0], wordGroup[jj][1], wordGroup[jj][2], wordGroup[jj][3]),
                                           (yP1, yP2, xP1, xP2)) == 1 and OA > th_OA:
                                    if wordGroup[jj][2] + wordGroup[jj][3] < xP1 + xP2 and stateWord == 0:
                                        delta = (wordGroup[jj][2] + wordGroup[jj][3]) / 2 - maskX1 - (th_OA - 0.5) * (
                                            wordGroup[jj][3] - wordGroup[jj][2])
                                        for kk in range(0, jj + 1):
                                            wordGroup[kk][2] -= delta
                                            wordGroup[kk][3] -= delta
                                        stateWord = 1
                                    else:
                                        delta = maskX2 - (wordGroup[jj][2] + wordGroup[jj][3]) / 2 - (th_OA - 0.5) * (
                                            wordGroup[jj][3] - wordGroup[jj][2])
                                        for kk in range(jj, len(inputString[layout[1][0][i][1]])):
                                            wordGroup[kk][2] += delta
                                            wordGroup[kk][3] += delta
                                        break
                            for jj in range(0, len(inputString[layout[1][0][i][1]])):
                                cGroupTemp.append(
                                    [wordGroup[jj][2], wordGroup[jj][0], inputString[layout[1][0][i][1]][jj], font.path,
                                     font.size, "up"])
                        else:
                            tempString = inputString[layout[1][0][i][1]]
                            lNum = int(width / height)
                            while (len(tempString) - len(tempString) % (lNum - 1)) / (lNum - 1) + 1 <= \
                                    layout[1][0][i][0][
                                        4]:
                                if lNum - len(tempString) % lNum > lNum - 1 - len(tempString) % (lNum - 1):
                                    lNum = lNum - 1
                                else:
                                    break
                            nL = int((len(tempString) - len(tempString) % lNum) / lNum) + 1
                            jStart = 0
                            if lTemp1.index(layout[1][0][i]) < lTemp1.index(layout[1][0][0]):
                                jStart = layout[1][0][i][0][4] - nL
                            for j in range(jStart, layout[1][0][i][0][4]):
                                t = ""
                                state = 0
                                if lNum < len(tempString):
                                    t = tempString[0:lNum]
                                    tempString = tempString[lNum:]
                                else:
                                    t = tempString
                                    state = 1
                                widthW = font.getsize(t)[0]
                                if n12 >= n11 and n12 >= n13:
                                    startP = 1 / 2 * (xT12 - xT11) - widthW / 2
                                elif n11 >= n12 and n11 >= n13:
                                    startP = 0
                                elif n13 >= n11 and n13 >= n12:
                                    startP = (xT12 - xT11) - widthW
                                y1 = yT11 + int(layout[1][0][i][0][0] * (yT12 - yT11)) + j * height
                                y2 = yT11 + int(layout[1][0][i][0][0] * (yT12 - yT11)) + int(height) + j * height
                                x1 = xT11 + int(startP)
                                x2 = x1 + font.getsize(t[0])[0]
                                maskX1 = 0
                                maskX2 = 0
                                for ii in range(1, len(mask[int((y1 + y2) / 2 - yP1)])):
                                    if mask[int((y1 + y2) / 2 - yP1)][ii] != 0 and mask[int((y1 + y2) / 2 - yP1)][
                                                ii - 1] == 0:
                                        maskX1 = xP1 + ii
                                    if mask[int((y1 + y2) / 2 - yP1)][ii] != 0 and mask[int((y1 + y2) / 2 - yP1)][
                                                ii + 1] == 0:
                                        maskX2 = xP1 + ii
                                        break
                                wordGroup = [[y1, y2, x1, x2]]
                                for jj in range(1, len(t)):
                                    x1 = x2
                                    x2 += font.getsize(t[jj])[0]
                                    wordGroup.append([y1, y2, x1, x2])
                                stateWord = 0
                                for jj in range(0, len(t)):
                                    OA = OverlapAreaMask(
                                        (wordGroup[jj][0], wordGroup[jj][1], wordGroup[jj][2], wordGroup[jj][3]),
                                        (yP1, yP2, xP1, xP2), mask)
                                    if Overlap((wordGroup[jj][0], wordGroup[jj][1], wordGroup[jj][2], wordGroup[jj][3]),
                                               (yP1, yP2, xP1, xP2)) == 1 and OA > th_OA:
                                        if wordGroup[jj][2] + wordGroup[jj][3] < xP1 + xP2 and stateWord == 0:
                                            delta = (wordGroup[jj][2] + wordGroup[jj][3]) / 2 - maskX1 - (
                                                                                                             th_OA - 0.5) * (
                                                                                                             wordGroup[
                                                                                                                 jj][
                                                                                                                 3] -
                                                                                                             wordGroup[
                                                                                                                 jj][2])
                                            for kk in range(0, jj + 1):
                                                wordGroup[kk][2] -= delta
                                                wordGroup[kk][3] -= delta
                                            stateWord = 1
                                        else:
                                            delta = maskX2 - (wordGroup[jj][2] + wordGroup[jj][3]) / 2 - (
                                                                                                             th_OA - 0.5) * (
                                                                                                             wordGroup[
                                                                                                                 jj][
                                                                                                                 3] -
                                                                                                             wordGroup[
                                                                                                                 jj][2])
                                            for kk in range(jj, len(t)):
                                                wordGroup[kk][2] += delta
                                                wordGroup[kk][3] += delta
                                            break
                                for jj in range(0, len(t)):
                                    cGroupTemp.append(
                                        [wordGroup[jj][2], wordGroup[jj][0], t[jj], font.path, font.size, "up"])
                                if state == 1:
                                    break
                    if state_Pic == 1 and layout[1][0][i] in downGroup:
                        downGroup.remove(layout[1][0][i])
                        if layout[1][0][i][0][4] == 1:
                            widthW = font.getsize(inputString[layout[1][0][i][1]])[0]
                            if n12 >= n11 and n12 >= n13:
                                startP = 1 / 2 * (xT12 - xT11) - widthW / 2
                            elif n11 >= n12 and n11 >= n13:
                                startP = 0
                            elif n13 >= n11 and n13 >= n12:
                                startP = (xT12 - xT11) - widthW
                            cGroupTemp.append([xT11 + int(startP), yT11 + int(layout[1][0][i][0][0] * (yT12 - yT11)),
                                               inputString[layout[1][0][i][1]], font.path, font.size, "down"])
                        else:
                            tempString = inputString[layout[1][0][i][1]]
                            lNum = int(width / height)
                            while (len(tempString) - len(tempString) % (lNum - 1)) / (lNum - 1) + 1 <= \
                                    layout[1][0][i][0][
                                        4]:
                                if lNum - len(tempString) % lNum > lNum - 1 - len(tempString) % (lNum - 1):
                                    lNum = lNum - 1
                                else:
                                    break
                            nL = int((len(tempString) - len(tempString) % lNum) / lNum) + 1
                            jStart = 0
                            if lTemp1.index(layout[1][0][i]) < lTemp1.index(layout[1][0][0]):
                                jStart = layout[1][0][i][0][4] - nL
                            for j in range(jStart, layout[1][0][i][0][4]):
                                t = ""
                                state = 0
                                if lNum < len(tempString):
                                    t = tempString[0:lNum]
                                    tempString = tempString[lNum:]
                                else:
                                    t = tempString
                                    state = 1
                                widthW = font.getsize(t)[0]
                                if n12 >= n11 and n12 >= n13:
                                    startP = 1 / 2 * (xT12 - xT11) - widthW / 2
                                elif n11 >= n12 and n11 >= n13:
                                    startP = 0
                                elif n13 >= n11 and n13 >= n12:
                                    startP = (xT12 - xT11) - widthW
                                cGroupTemp.append([xT11 + int(startP),
                                                   yT11 + int(layout[1][0][i][0][0] * (yT12 - yT11)) + int(j * height),
                                                   t, font.path, font.size, "down"])
                                if state == 1:
                                    break

                for i in range(0, len(lTemp2)):
                    width = int((layout[1][1][i][0][3] - layout[1][1][i][0][2]) * (xT22 - xT21))
                    height = int(
                        (layout[1][1][i][0][1] - layout[1][1][i][0][0]) * (yT22 - yT21) / layout[1][1][i][0][4])
                    rr = random.randint(1, 3)
                    if layout[1][1][i][1] > 3:
                        font = ImageFont.truetype(path_font + str(3) + '/' + str(rr) + ".ttf", height, encoding="unic")
                    else:
                        font = ImageFont.truetype(path_font + str(layout[1][1][i][1]) + '/' + str(rr) + ".ttf", height,
                                                  encoding="unic")
                    if layout[1][1][i] in upGroup:
                        upGroup.remove(layout[1][1][i])
                        if layout[1][1][i][0][4] == 1:
                            widtW = font.getsize(inputString[layout[1][1][i][1]])[0]
                            if n22 >= n21 and n22 >= n23:
                                startP = 1 / 2 * (xT22 - xT21) - widtW / 2
                            elif n21 >= n22 and n21 >= n23:
                                startP = 0
                            elif n23 >= n21 and n23 >= n22:
                                startP = (xT22 - xT21) - widtW
                            y1 = yT21 + int(layout[1][1][i][0][0] * (yT22 - yT21))
                            y2 = yT21 + int(layout[1][1][i][0][0] * (yT22 - yT21)) + int(height)
                            x1 = xT21 + int(startP)
                            x2 = x1 + font.getsize(inputString[layout[1][1][i][1]][0])[0]
                            maskX1 = 0
                            maskX2 = 0
                            for ii in range(1, len(mask[int((y1 + y2) / 2 - yP1)])):
                                if mask[int((y1 + y2) / 2 - yP1)][ii] != 0 and mask[int((y1 + y2) / 2 - yP1)][
                                            ii - 1] == 0:
                                    maskX1 = xP1 + ii
                                if mask[int((y1 + y2) / 2 - yP1)][ii] != 0 and mask[int((y1 + y2) / 2 - yP1)][
                                            ii + 1] == 0:
                                    maskX2 = xP1 + ii
                                    break
                            wordGroup = [[y1, y2, x1, x2]]
                            for jj in range(1, len(inputString[layout[1][1][i][1]])):
                                x1 = x2
                                x2 += font.getsize(inputString[layout[1][1][i][1]][jj])[0]
                                wordGroup.append([y1, y2, x1, x2])
                            stateWord = 0
                            for jj in range(0, len(inputString[layout[1][1][i][1]])):
                                OA = OverlapAreaMask(
                                    (wordGroup[jj][0], wordGroup[jj][1], wordGroup[jj][2], wordGroup[jj][3]),
                                    (yP1, yP2, xP1, xP2), mask)
                                if Overlap((wordGroup[jj][0], wordGroup[jj][1], wordGroup[jj][2], wordGroup[jj][3]),
                                           (yP1, yP2, xP1, xP2)) == 1 and OA > th_OA:
                                    if wordGroup[jj][2] + wordGroup[jj][3] < xP1 + xP2 and stateWord == 0:
                                        delta = (wordGroup[jj][2] + wordGroup[jj][3]) / 2 - maskX1 - (th_OA - 0.5) * (
                                            wordGroup[jj][3] - wordGroup[jj][2])
                                        for kk in range(0, jj + 1):
                                            wordGroup[kk][2] -= delta
                                            wordGroup[kk][3] -= delta
                                        stateWord = 1
                                    else:
                                        delta = maskX2 - (wordGroup[jj][2] + wordGroup[jj][3]) / 2 - (th_OA - 0.5) * (
                                            wordGroup[jj][3] - wordGroup[jj][2])
                                        for kk in range(jj, len(inputString[layout[1][1][i][1]])):
                                            wordGroup[kk][2] += delta
                                            wordGroup[kk][3] += delta
                                        break
                            for jj in range(0, len(inputString[layout[1][1][i][1]])):
                                cGroupTemp.append(
                                    [wordGroup[jj][2], wordGroup[jj][0], inputString[layout[1][1][i][1]][jj], font.path,
                                     font.size, "up"])
                        else:
                            tempString = inputString[layout[1][1][i][1]]
                            lNum = int(width / height)
                            while (len(tempString) - len(tempString) % (lNum - 1)) / (lNum - 1) + 1 <= \
                                    layout[1][1][i][0][
                                        4]:
                                if lNum - len(tempString) % lNum > lNum - 1 - len(tempString) % (lNum - 1):
                                    lNum = lNum - 1
                                else:
                                    break
                            nL = int((len(tempString) - len(tempString) % lNum) / lNum) + 1
                            jStart = 0
                            if lTemp2.index(layout[1][1][i]) < lTemp2.index(layout[1][1][0]):
                                jStart = layout[1][1][i][0][4] - nL
                            for j in range(jStart, layout[1][1][i][0][4]):
                                t = ""
                                state = 0
                                if lNum < len(tempString):
                                    t = tempString[0:lNum]
                                    tempString = tempString[lNum:]
                                else:
                                    t = tempString
                                    state = 1
                                widthW = font.getsize(t)[0]
                                if n22 >= n21 and n22 >= n23:
                                    startP = 1 / 2 * (xT22 - xT21) - widthW / 2
                                elif n21 >= n22 and n21 >= n23:
                                    startP = 0
                                elif n23 >= n21 and n23 >= n22:
                                    startP = (xT22 - xT21) - widthW
                                y1 = yT21 + int(layout[1][1][i][0][0] * (yT22 - yT21)) + j * height
                                y2 = yT21 + int(layout[1][1][i][0][0] * (yT22 - yT21)) + int(height) + j * height
                                x1 = xT21 + int(startP)
                                x2 = x1 + font.getsize(t[0])[0]
                                maskX1 = 0
                                maskX2 = 0
                                for ii in range(1, len(mask[int((y1 + y2) / 2 - yP1)])):
                                    if mask[int((y1 + y2) / 2 - yP1)][ii] != 0 and mask[int((y1 + y2) / 2 - yP1)][
                                                ii - 1] == 0:
                                        maskX1 = xP1 + ii
                                    if mask[int((y1 + y2) / 2 - yP1)][ii] != 0 and mask[int((y1 + y2) / 2 - yP1)][
                                                ii + 1] == 0:
                                        maskX2 = xP1 + ii
                                        break
                                wordGroup = [[y1, y2, x1, x2]]
                                for jj in range(1, len(t)):
                                    x1 = x2
                                    x2 += font.getsize(t[jj])[0]
                                    wordGroup.append([y1, y2, x1, x2])
                                stateWord = 0
                                for jj in range(0, len(t)):
                                    OA = OverlapAreaMask(
                                        (wordGroup[jj][0], wordGroup[jj][1], wordGroup[jj][2], wordGroup[jj][3]),
                                        (yP1, yP2, xP1, xP2), mask)
                                    if Overlap((wordGroup[jj][0], wordGroup[jj][1], wordGroup[jj][2], wordGroup[jj][3]),
                                               (yP1, yP2, xP1, xP2)) == 1 and OA > th_OA:
                                        if wordGroup[jj][2] + wordGroup[jj][3] < xP1 + xP2 and stateWord == 0:
                                            delta = (wordGroup[jj][2] + wordGroup[jj][3]) / 2 - maskX1 - (
                                                                                                             th_OA - 0.5) * (
                                                                                                             wordGroup[
                                                                                                                 jj][
                                                                                                                 3] -
                                                                                                             wordGroup[
                                                                                                                 jj][
                                                                                                                 2])
                                            for kk in range(0, jj + 1):
                                                wordGroup[kk][2] -= delta
                                                wordGroup[kk][3] -= delta
                                            stateWord = 1
                                        else:
                                            delta = maskX2 - (wordGroup[jj][2] + wordGroup[jj][3]) / 2 - (
                                                                                                             th_OA - 0.5) * (
                                                                                                             wordGroup[
                                                                                                                 jj][
                                                                                                                 3] -
                                                                                                             wordGroup[
                                                                                                                 jj][2])
                                            for kk in range(jj, len(t)):
                                                wordGroup[kk][2] += delta
                                                wordGroup[kk][3] += delta
                                            break
                                for jj in range(0, len(t)):
                                    cGroupTemp.append(
                                        [wordGroup[jj][2], wordGroup[jj][0], t[jj], font.path, font.size, "up"])
                                if state == 1:
                                    break
                    if len(upGroup) == 0 and state_Pic == 0:
                        state_Pic = 1
                    if state_Pic == 1 and layout[1][1][i] in downGroup:
                        downGroup.remove(layout[1][1][i])
                        if layout[1][1][i][0][4] == 1:
                            widtW = font.getsize(inputString[layout[1][1][i][1]])[0]
                            if n22 >= n21 and n22 >= n23:
                                startP = 1 / 2 * (xT22 - xT21) - widtW / 2
                            elif n21 >= n22 and n21 >= n23:
                                startP = 0
                            elif n23 >= n21 and n23 >= n22:
                                startP = (xT22 - xT21) - widtW
                            cGroupTemp.append([xT21 + int(startP), yT21 + int(layout[1][1][i][0][0] * (yT22 - yT21)),
                                               inputString[layout[1][1][i][1]], font.path, font.size, "down"])
                        else:
                            tempString = inputString[layout[1][1][i][1]]
                            lNum = int(width / height)
                            while (len(tempString) - len(tempString) % (lNum - 1)) / (lNum - 1) + 1 <= \
                                    layout[1][1][i][0][
                                        4]:
                                if lNum - len(tempString) % lNum > lNum - 1 - len(tempString) % (lNum - 1):
                                    lNum = lNum - 1
                                else:
                                    break
                            nL = int((len(tempString) - len(tempString) % lNum) / lNum) + 1
                            jStart = 0
                            if lTemp2.index(layout[1][1][i]) < lTemp2.index(layout[1][1][0]):
                                jStart = layout[1][1][i][0][4] - nL
                            for j in range(jStart, layout[1][1][i][0][4]):
                                t = ""
                                state = 0
                                if lNum < len(tempString):
                                    t = tempString[0:lNum]
                                    tempString = tempString[lNum:]
                                else:
                                    t = tempString
                                    state = 1
                                widthW = font.getsize(t)[0]
                                if n22 >= n21 and n22 >= n23:
                                    startP = 1 / 2 * (xT22 - xT21) - widthW / 2
                                elif n21 >= n22 and n21 >= n23:
                                    startP = 0
                                elif n23 >= n21 and n23 >= n22:
                                    startP = (xT22 - xT21) - widthW
                                cGroupTemp.append([xT21 + int(startP),
                                                   yT21 + int(layout[1][1][i][0][0] * (yT22 - yT21)) + int(j * height),
                                                   t, font.path, font.size, "down"])
                                if state == 1:
                                    break

            classGroupText.append(cGroupTemp)
            if pic_number == 9:
                break
            pic_number += 1
        classGroup.append(classGroupText)
        finalGroup.append(classGroup)
    with open(path_Out, 'wb') as handle:
        pickle.dump(finalGroup, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path_Out


def PictureRecovery(path_Data, color, path_Background, path_Product):
    with open(path_Data, 'rb') as handle:
        InputList = pickle.load(handle)
    product = Image.open(path_Product)
    alphaBackground = 128
    alphaForeground = 255
    imgCollection = []
    for imgClass in InputList:
        product1 = product.resize((int(product.size[0] * (imgClass[0][1] - imgClass[0][0]) / product.size[1]),
                                   imgClass[0][1] - imgClass[0][0]))
        r, g, b, a = product1.split()
        imgCollectionTemp = []
        for group in imgClass[1]:
            background = Image.open(path_Background)
            draw = ImageDraw.Draw(background)
            for each in group:
                if each[5] == "up":
                    font = ImageFont.truetype(each[3], each[4], encoding="unic")
                    draw.text((each[0], each[1]), each[2], fill=(color[0], color[1], color[2], alphaBackground),
                              font=font)
            background.paste(product1, (imgClass[0][2], imgClass[0][0], imgClass[0][3], imgClass[0][1]), mask=a)
            for each in group:
                if each[5] == "down":
                    font = ImageFont.truetype(each[3], each[4], encoding="unic")
                    draw.text((each[0], each[1]), each[2], fill=(color[0], color[1], color[2], alphaForeground),
                              font=font)
            imgCollectionTemp.append(background)
        imgCollection.append(imgCollectionTemp)
    return imgCollection


path_In_CSV = "G:/浙大实习/数据label/JsonDataForBanner.csv"
path_In_AllPic = "G:/浙大实习/allImages-8337/"
path_In_Text = "G:/浙大实习/text聚类/"
path_In_SourceText = "G:/浙大实习/text_product聚类/One2TwoText.txt"
path_In_SourceProduct = "G:/浙大实习/text_product聚类/One2TwoProduct.txt"
path_Font = "G:/Font/"
path_In_ChosenProduct = path_Font + "1.png"
path_In_ChosenBackground = path_Font + "2.png"
inputString = ["优雅时尚气质款", "新品男女鞋包6折起", "New Arrival", "TIME: 3月21日-3月23日"]
path_Out = "G:/Font/positionTwo.pickle"
hP = 700
wP = 1920
output = LayoutTwo(path_In_CSV, path_In_AllPic, path_In_Text, path_In_SourceText, path_In_SourceProduct, path_Font,
                   path_In_ChosenProduct, inputString, hP, wP,path_Out)
pic = PictureRecovery(output, (192, 128, 114), path_In_ChosenBackground, path_In_ChosenProduct)
x = 1
