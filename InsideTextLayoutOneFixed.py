import numpy as np
import os, os.path, shutil
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import csv
import json
import copy
import random

global sinputString, hP, wP
inputString = ["天猫新风尚", "钜惠来袭", "用红包享满减全场包邮", "325开启仅限大陆地区"]
# inputString = ["优雅时尚气质款", "新品男女鞋包6折起", "New Arrival", "TIME: 3月21日-3月23日"]
hP = Image.open("G:/Font/2.png").size[1]
wP = Image.open("G:/Font/2.png").size[0]


def ReadDataOne(numC):
    path_S = "G:/浙大实习/text_product聚类/One2OneText.txt"
    fs = open(path_S, 'r').readlines()
    tp = fs[numC + 1][:-1].split(' ')
    hT = float(tp[2])
    wT = float(tp[3])
    TextBox = []
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
                TextTemp = [name]
                for each in item['texts']:
                    x = each['x']
                    y = each['y']
                    width = each['width']
                    height = each['height']
                    line = each['lines']
                    TextTemp.append([y, y + height, x, x + width, line])
                TextBox.append(TextTemp)
    pathTextOut = "G:/浙大实习/text聚类/"
    collection = []
    dirTOut = os.listdir(pathTextOut)
    thT = 0.6
    for each in dirTOut:
        pic_name = each[:-4] + ".jpg"
        tempC = [pic_name]
        fOut = open(pathTextOut + each, 'r').readlines()
        if len(fOut) - 1 == 1:
            if fOut[1][len(fOut[1]) - 1] == '\n':
                fOut[1] = fOut[1][:-1]
            strO = fOut[1].split(' ')
            for i in range(0, len(strO)):
                strO[i] = int(strO[i])
            h1 = strO[1] - strO[0]
            w1 = strO[3] - strO[2]
            if thT < (h1 / w1) / (hT / wT) < 1 / thT:
                tempA = []
                for aa in TextBox:
                    state = 0
                    if aa[0] == pic_name:
                        state = 1
                        k = 0
                        for bb in aa:
                            if k != 0:
                                tempA.append(
                                    [(bb[0] - strO[0]) / (strO[1] - strO[0]), (bb[1] - strO[0]) / (strO[1] - strO[0]),
                                     (bb[2] - strO[2]) / (strO[3] - strO[2]), (bb[3] - strO[2]) / (strO[3] - strO[2]),
                                     int(bb[4])])
                            k += 1
                    if state == 1:
                        break
                tempA.sort(key=lambda x: (x[1] - x[0]) / x[4], reverse=True)
                tempC.append(tempA)
                collection.append(tempC)
    return collection


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


def SortInput(x):
    temp = []
    k = 0
    for each in x[1]:
        if each[4] == 1:
            temp.append(0)
        else:
            width = int((each[3] - each[2]) * wP)
            height = int((each[1] - each[0]) / each[4] * hP)
            n = int(width / height)
            x1 = len(inputString[k]) % n
            x2 = (len(inputString[k]) - x1) / n
            if x1 != 0:
                temp.append(each[4] - (x2 + 1))
            else:
                temp.append(each[4] - x2)
        k += 1
    return sum(temp)


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


def OverlapArea(r1, r2):
    y1 = max(r1[0], r2[0])
    y2 = min(r1[1], r2[1])
    x1 = max(r1[2], r2[2])
    x2 = min(r1[3], r2[3])
    return (y2 - y1) * (x2 - x1) / ((r1[1] - r1[0]) * (r1[3] - r1[2]))


def OverlapAreaMask(r1, mask):
    count = 0
    for i in range(int(r1[0]), int(r1[1])):
        for j in range(int(r1[2]), int(r1[3])):
            if yP1 < i < yP2 and xP1 < j < xP2 and mask[i - yP1][j - xP1] != 0:
                count += 1
    return count / ((r1[1] - r1[0]) * (r1[3] - r1[2]))


path_des = "G:/浙大实习/text内部聚类/LayoutOne1Ratio/"
if os.path.exists(path_des):
    shutil.rmtree(path_des)
os.mkdir(path_des)
path_P = "G:/浙大实习/text_product聚类/One2OneProduct.txt"
path_T = "G:/浙大实习/text_product聚类/One2OneText.txt"
path_font = "G:/Font/"
for classNum in range(0, 6):
    fP = open(path_P, 'r').readlines()[classNum + 1]
    fT = open(path_T, 'r').readlines()[classNum + 1]
    yP1 = int((float(fP.split(' ')[0]) - 0.5 * float(fP.split(' ')[2])) * hP)
    yP2 = int((float(fP.split(' ')[0]) + 0.5 * float(fP.split(' ')[2])) * hP)
    xP1 = int((float(fP.split(' ')[1]) - 0.5 * float(fP.split(' ')[3])) * wP)
    xP2 = int((float(fP.split(' ')[1]) + 0.5 * float(fP.split(' ')[3])) * wP)
    yT1 = int((float(fP.split(' ')[0]) + float(fT.split(' ')[0]) - 0.5 * float(fT.split(' ')[2])) * hP)
    yT2 = int((float(fP.split(' ')[0]) + float(fT.split(' ')[0]) + 0.5 * float(fT.split(' ')[2])) * hP)
    xT1 = int((float(fP.split(' ')[1]) + float(fT.split(' ')[1]) - 0.5 * float(fT.split(' ')[3])) * wP)
    xT2 = int((float(fP.split(' ')[1]) + float(fT.split(' ')[1]) + 0.5 * float(fT.split(' ')[3])) * wP)
    TotalData = ReadDataOne(classNum)
    path_c = path_des + str(classNum) + '/'
    os.mkdir(path_c)

    th = 0.3
    TextGroup = []
    for data in TotalData:
        pic_name = data[0]
        lenGroup = [[data[1][0][0], data[1][0][1]]]
        tGroup = [data[1][0]]
        for each in data[1][1:]:
            if LineOverlap(lenGroup, [each[0], each[1]]) < th:
                lenGroup.append([each[0], each[1]])
                tGroup.append(each)
        TextGroup.append([pic_name, tGroup])

    num = len(inputString)
    TargetTextGroup = []
    for each in TextGroup:
        if len(each[1]) == num:
            TargetTextGroup.append(each)

    SuitableTextGroup = []
    for each in TargetTextGroup:
        k = 0
        state = 0
        for tt in each[1]:
            if int((tt[3] - tt[2]) * (xT2 - xT1) / ((tt[1] - tt[0]) * (yT2 - yT1) / tt[4])) * tt[4] >= len(
                    inputString[k]):
                state += 1
            k += 1
        if state == num:
            SuitableTextGroup.append(each)
    SuitableTextGroup.sort(key=SortInput)

    count = 0
    th_d = 0.2
    th_h_r = 1.5
    pic_number = 0
    for layout in SuitableTextGroup:
        lTemp = copy.deepcopy(layout[1])
        lTemp.sort()
        mapBox = []
        for each in lTemp:
            mapBox.append(layout[1].index(each))
        for i in np.linspace(0, num - 2, num - 1):
            i = int(i)
            if lTemp[i][1] > lTemp[i + 1][0] or lTemp[i + 1][0] - lTemp[i][1] < (
                        lTemp[i + 1][1] - lTemp[i + 1][0]) * th_d:
                if (lTemp[i][1] - lTemp[i][0]) / (lTemp[i + 1][1] - lTemp[i + 1][0]) > th_h_r:
                    lTemp[i][1] = lTemp[i + 1][0] - (lTemp[i + 1][1] - lTemp[i + 1][0]) * th_d
                elif (lTemp[i + 1][1] - lTemp[i + 1][0]) / (lTemp[i][1] - lTemp[i][0]) > th_h_r:
                    lTemp[i + 1][0] = lTemp[i][1] + (lTemp[i][1] - lTemp[i][0]) * th_d
                else:
                    ratio = (lTemp[i][1] - lTemp[i][0]) / (
                        lTemp[i + 1][1] - lTemp[i + 1][0] + lTemp[i][1] - lTemp[i][0])
                    dd = 0
                    if (lTemp[i + 1][1] - lTemp[i + 1][0]) <= (lTemp[i][1] - lTemp[i][0]):
                        dd = (lTemp[i + 1][1] - lTemp[i + 1][0]) * th_d + lTemp[i][1] - lTemp[i + 1][0]
                    else:
                        dd = (lTemp[i][1] - lTemp[i][0]) * th_d + lTemp[i][1] - lTemp[i + 1][0]
                    lTemp[i][1] = lTemp[i][1] - ratio * dd
                    lTemp[i + 1][0] = lTemp[i + 1][0] + (1 - ratio) * dd
        for i in np.linspace(0, num - 1, num):
            i = int(i)
            layout[1][mapBox[i]] = lTemp[i]
        layout[1].sort(key=lambda x: (x[1] - x[0]) / x[4], reverse=True)
        image = Image.open(path_font + "2.png")
        # image = Image.fromarray(array)
        draw = ImageDraw.Draw(image)

        imp = Image.open(path_font + "1.png")
        imp1 = imp.resize((int(imp.size[0] * (yP2 - yP1) / imp.size[1]), yP2 - yP1))
        xP1Temp = int((xP1 + xP2 - int(imp.size[0] * (yP2 - yP1) / imp.size[1])) / 2)
        xP2Temp = int((xP1 + xP2 + int(imp.size[0] * (yP2 - yP1) / imp.size[1])) / 2)
        xP1 = xP1Temp
        xP2 = xP2Temp
        r, g, b, a = imp1.split()
        mask = np.array(a)
        mask = mask.tolist()

        tagBox = []
        for i in np.linspace(0, num - 1, num):
            i = int(i)
            if i != 0:
                middleD = ((layout[1][i][2] + layout[1][i][3]) / 2 - (layout[1][0][2] + layout[1][0][3]) / 2) / (
                    layout[1][i][3] - layout[1][i][2])
                if middleD <= -1 / 4:
                    tagBox.append(-1)
                elif middleD >= 1 / 4:
                    tagBox.append(1)
                else:
                    tagBox.append(0)
        n1 = tagBox.count(-1)
        n2 = tagBox.count(0)
        n3 = tagBox.count(1)

        upGroup = []
        downGroup = []
        for i in range(0, num):
            yy = yT1 + int(layout[1][i][0] * (yT2 - yT1) - (layout[1][i][1] - layout[1][i][0]) / layout[1][i][4] * (
                yT2 - yT1) * 0.2 + (layout[1][i][1] - layout[1][i][0]) * (yT2 - yT1) * 0.5)
            xx = 0
            if n2 >= n1 and n2 >= n3:
                xx = 1 / 2 * (xT2 - xT1) + xT1
            elif n1 >= n2 and n1 >= n3:
                xx = len(inputString[i]) / 2 * (yT2 - yT1) / layout[1][i][4] + xT1
            elif n3 >= n1 and n3 >= n2:
                xx = (xT2 - xT1) - len(inputString[i]) / 2 * (yT2 - yT1) / layout[1][i][4] + xT1
            if yy < hP / 2:
                upGroup.append(layout[1][i])
            else:
                downGroup.append(layout[1][i])

        state_Pic = 0
        th_OA = 0.5
        while len(upGroup) != 0 or len(downGroup) != 0:
            for i in range(0, num):
                width = int((layout[1][i][3] - layout[1][i][2]) * (xT2 - xT1))
                height = int((layout[1][i][1] - layout[1][i][0]) * (yT2 - yT1) / layout[1][i][4])
                rr = random.randint(1, 3)
                if i > 3:
                    font = ImageFont.truetype(path_font + str(3) + '/' + str(rr) + ".ttf", height, encoding="unic")
                else:
                    font = ImageFont.truetype(path_font + str(i) + '/' + str(rr) + ".ttf", height, encoding="unic")
                if layout[1][i] in upGroup:
                    upGroup.remove(layout[1][i])
                    if layout[1][i][4] == 1:
                        widthW = font.getsize(inputString[i])[0]
                        if n2 >= n1 and n2 >= n3:
                            startP = 1 / 2 * (xT2 - xT1) - widthW / 2
                        elif n1 >= n2 and n1 >= n3:
                            startP = 0
                        elif n3 >= n1 and n3 >= n2:
                            startP = (xT2 - xT1) - widthW
                        y1 = yT1 + int(layout[1][i][0] * (yT2 - yT1))
                        y2 = yT1 + int(layout[1][i][0] * (yT2 - yT1)) + int(height)
                        x1 = xT1 + int(startP)
                        x2 = x1 + font.getsize(inputString[i][0])[0]
                        maskX1 = 0
                        maskX2 = 0
                        for ii in range(1, len(mask[int((y1 + y2) / 2 - yP1)])):
                            if mask[int((y1 + y2) / 2 - yP1)][ii] != 0 and mask[int((y1 + y2) / 2 - yP1)][ii - 1] == 0:
                                maskX1 = xP1 + ii
                            if mask[int((y1 + y2) / 2 - yP1)][ii] != 0 and mask[int((y1 + y2) / 2 - yP1)][ii + 1] == 0:
                                maskX2 = xP1 + ii
                                break
                        wordGroup = [[y1, y2, x1, x2]]
                        for jj in range(1, len(inputString[i])):
                            x1 = x2
                            x2 += font.getsize(inputString[i][jj])[0]
                            wordGroup.append([y1, y2, x1, x2])
                        stateWord = 0
                        for jj in range(0, len(inputString[i])):
                            OA = OverlapAreaMask(
                                (wordGroup[jj][0], wordGroup[jj][1], wordGroup[jj][2], wordGroup[jj][3]), mask)
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
                                    for kk in range(jj, len(inputString[i])):
                                        wordGroup[kk][2] += delta
                                        wordGroup[kk][3] += delta
                                    break
                        for jj in range(0, len(inputString[i])):
                            draw.text((wordGroup[jj][2], wordGroup[jj][0]), inputString[i][jj], 'black', font)
                    else:
                        tempString = inputString[i]
                        lNum = int(width / height)
                        while (len(tempString) - len(tempString) % (lNum - 1)) / (lNum - 1) + 1 <= layout[1][i][4]:
                            if lNum - len(tempString) % lNum > lNum - 1 - len(tempString) % (lNum - 1):
                                lNum = lNum - 1
                            else:
                                break
                        nL = int((len(tempString) - len(tempString) % lNum) / lNum) + 1
                        jStart = 0
                        if lTemp.index(layout[1][i]) < lTemp.index(layout[1][0]):
                            jStart = layout[1][i][4] - nL
                        for j in range(jStart, layout[1][i][4]):
                            t = ""
                            state = 0
                            if lNum < len(tempString):
                                t = tempString[0:lNum]
                                tempString = tempString[lNum:]
                            else:
                                t = tempString
                                state = 1
                            widthW = font.getsize(t)[0]
                            if n2 >= n1 and n2 >= n3:
                                startP = 1 / 2 * (xT2 - xT1) - widthW / 2
                            elif n1 >= n2 and n1 >= n3:
                                startP = 0
                            elif n3 >= n1 and n3 >= n2:
                                startP = (xT2 - xT1) - widthW
                            y1 = yT1 + int(layout[1][i][0] * (yT2 - yT1)) + j * height
                            y2 = yT1 + int(layout[1][i][0] * (yT2 - yT1)) + j * height + int(height)
                            x1 = xT1 + int(startP)
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
                                    (wordGroup[jj][0], wordGroup[jj][1], wordGroup[jj][2], wordGroup[jj][3]), mask)
                                if Overlap((wordGroup[jj][0], wordGroup[jj][1], wordGroup[jj][2], wordGroup[jj][3]),
                                           (yP1, yP2, xP1, xP2)) == 1 and OA > th_OA:
                                    if wordGroup[jj][2] + wordGroup[jj][3] < xP1 + xP2 and stateWord == 0:
                                        delta = (wordGroup[jj][2] + wordGroup[jj][3]) / 2 - maskX1 - (th_OA - 0.5) * (
                                            wordGroup[jj][3] - wordGroup[jj][2])
                                        for kk in range(0, len(t)):
                                            wordGroup[kk][2] -= delta
                                            wordGroup[kk][3] -= delta
                                        stateWord = 1
                                    else:
                                        delta = maskX2 - (wordGroup[jj][2] + wordGroup[jj][3]) / 2 - (th_OA - 0.5) * (
                                            wordGroup[jj][3] - wordGroup[jj][2])
                                        for kk in range(jj, len(t)):
                                            wordGroup[kk][2] += delta
                                            wordGroup[kk][3] += delta
                                        break
                            for jj in range(0, len(t)):
                                draw.text((wordGroup[jj][2], wordGroup[jj][0]), t[jj], 'black', font)
                            if state == 1:
                                break
                if len(upGroup) == 0 and state_Pic == 0:
                    image.paste(imp1, (xP1, yP1, xP2, yP2), mask=a)
                    state_Pic = 1
                if state_Pic == 1 and layout[1][i] in downGroup:
                    downGroup.remove(layout[1][i])
                    if layout[1][i][4] == 1:
                        widthW = font.getsize(inputString[i])[0]
                        if n2 >= n1 and n2 >= n3:
                            startP = 1 / 2 * (xT2 - xT1) - widthW / 2
                        elif n1 >= n2 and n1 >= n3:
                            startP = 0
                        elif n3 >= n1 and n3 >= n2:
                            startP = (xT2 - xT1) - widthW
                        draw.text((xT1 + int(startP), yT1 + int(layout[1][i][0] * (yT2 - yT1))), inputString[i],
                                  'black', font)
                    else:
                        tempString = inputString[i]
                        lNum = int(width / height)
                        while (len(tempString) - len(tempString) % (lNum - 1)) / (lNum - 1) + 1 <= layout[1][i][4]:
                            if lNum - len(tempString) % lNum > lNum - 1 - len(tempString) % (lNum - 1):
                                lNum = lNum - 1
                            else:
                                break
                        nL = int((len(tempString) - len(tempString) % lNum) / lNum) + 1
                        jStart = 0
                        if lTemp.index(layout[1][i]) < lTemp.index(layout[1][0]):
                            jStart = layout[1][i][4] - nL
                        for j in range(jStart, layout[1][i][4]):
                            t = ""
                            state = 0
                            if lNum < len(tempString):
                                t = tempString[0:lNum]
                                tempString = tempString[lNum:]
                            else:
                                t = tempString
                                state = 1
                            widthW = font.getsize(t)[0]
                            if n2 >= n1 and n2 >= n3:
                                startP = 1 / 2 * (xT2 - xT1) - widthW / 2
                            elif n1 >= n2 and n1 >= n3:
                                startP = 0
                            elif n3 >= n1 and n3 >= n2:
                                startP = (xT2 - xT1) - widthW
                            draw.text((xT1 + int(startP), yT1 + int(layout[1][i][0] * (yT2 - yT1)) + int(j * height)),
                                      t, 'black', font)
                            if state == 1:
                                break
        image.save(path_c + str(count) + "_" + layout[0][:-4] + ".png")
        count += 1

        if pic_number == 9:
            break
        pic_number += 1
