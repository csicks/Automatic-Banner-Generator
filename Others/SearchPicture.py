import numpy as np
import os, os.path, shutil
from PIL import Image


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
    return (y2 - y1) * (x2 - x1)


type0 = 2
pathG_pic = "G:/浙大实习/聚类结果/"
path_Text = "G:/浙大实习/text聚类/"
path_des = "G:/浙大实习/text_product聚类/"
dirT = os.listdir(path_Text)

if type0 == 1:
    picI_path = "G:/浙大实习/过程文件/结果1/"
    if os.path.exists(path_des + 'SelectOne'):
        shutil.rmtree(path_des + 'SelectOne')
    os.mkdir(path_des + 'SelectOne')
    th = 0.7
    dirP = os.listdir(picI_path)
    for pp in dirP:
        SPGroup = []
        num = int(pp[0]) + 1
        pathSource = "G:/浙大实习/text_product聚类/One2OneText.txt"
        path_Product = "G:/浙大实习/text_product聚类/One2OneProduct.txt"
        k = 0
        picIShape = []
        for line in open(pathSource, 'r'):
            if k == num:
                px = 0
                py = 0
                fP = open(path_Product, 'r').readlines()
                kP = 0
                for lineP in fP:
                    if kP == num:
                        if lineP[len(lineP) - 1] == '\n':
                            lineP = lineP[:-1]
                        py = float(lineP.split(' ')[0])
                        px = float(lineP.split(' ')[1])
                    kP += 1
                if line[len(line) - 1] == '\n':
                    line = line[:-1]
                y1 = float(line.split(' ')[0]) + py - 0.5 * float(line.split(' ')[2])
                y2 = float(line.split(' ')[0]) + py + 0.5 * float(line.split(' ')[2])
                x1 = float(line.split(' ')[1]) + px - 0.5 * float(line.split(' ')[3])
                x2 = float(line.split(' ')[1]) + px + 0.5 * float(line.split(' ')[3])
                picIShape = [y1, y2, x1, x2]
                break
            k += 1
        picSShape = []
        for each in dirT:
            path1 = path_Text + each
            f = open(path1, 'r').readlines()
            n1 = len(f) - 1
            img = Image.open(pathG_pic + each[:-4] + '.jpg')
            shape = img.size
            if n1 == type0:
                picSShape = [int(f[1].split(' ')[0]) / shape[1], int(f[1].split(' ')[1]) / shape[1],
                             int(f[1].split(' ')[2]) / shape[0], int(f[1].split(' ')[3]) / shape[0]]
                OArea = OverlapArea(picIShape, picSShape)
                SArea = (picSShape[1] - picSShape[0]) * (picSShape[3] - picSShape[2])
                IArea = (picIShape[1] - picIShape[0]) * (picIShape[3] - picIShape[2])
                if Overlap(picSShape, picIShape) == 1 and OArea / SArea >= th and OArea / IArea >= th:
                    SPGroup.append(each[:-4] + '.jpg')
        path2 = path_des + 'SelectOne/' + str(num - 1)
        if os.path.exists(path2):
            shutil.rmtree(path2)
        os.mkdir(path2)
        for each in SPGroup:
            shutil.copyfile(pathG_pic + each, path2 + '/' + each)

elif type0 == 2:
    picI_path = "G:/浙大实习/过程文件/结果2/"
    if os.path.exists(path_des + 'SelectTwo'):
        shutil.rmtree(path_des + 'SelectTwo')
    os.mkdir(path_des + 'SelectTwo')
    th = 0.7
    dirP = os.listdir(picI_path)
    for pp in dirP:
        SPGroup = []
        num = int(pp[0]) + 1
        pathSource = "G:/浙大实习/text_product聚类/One2TwoText.txt"
        path_Product = "G:/浙大实习/text_product聚类/One2TwoProduct.txt"
        k = 0
        picIShape = []
        for line in open(pathSource, 'r'):
            if k == num:
                px = 0
                py = 0
                fP = open(path_Product, 'r').readlines()
                kP = 0
                for lineP in fP:
                    if kP == num:
                        if lineP[len(lineP) - 1] == '\n':
                            lineP = lineP[:-1]
                        py = float(lineP.split(' ')[0])
                        px = float(lineP.split(' ')[1])
                    kP += 1
                if line[len(line) - 1] == '\n':
                    line = line[:-1]
                y11 = float(line.split(' ')[0]) + py - 0.5 * float(line.split(' ')[2])
                y12 = float(line.split(' ')[0]) + py + 0.5 * float(line.split(' ')[2])
                x11 = float(line.split(' ')[1]) + px - 0.5 * float(line.split(' ')[3])
                x12 = float(line.split(' ')[1]) + px + 0.5 * float(line.split(' ')[3])
                y21 = float(line.split(' ')[4]) + py - 0.5 * float(line.split(' ')[6])
                y22 = float(line.split(' ')[4]) + py + 0.5 * float(line.split(' ')[6])
                x21 = float(line.split(' ')[5]) + px - 0.5 * float(line.split(' ')[7])
                x22 = float(line.split(' ')[5]) + px + 0.5 * float(line.split(' ')[7])
                picIShape = [[y11, y12, x11, x12], [y21, y22, x21, x22]]
                break
            k += 1
        picSShape = []
        for each in dirT:
            path1 = path_Text + each
            f = open(path1, 'r').readlines()
            n1 = len(f) - 1
            img = Image.open(pathG_pic + each[:-4] + '.jpg')
            shape = img.size
            if n1 == type0:
                xxx = f[1].split(' ')
                picSShape1 = [int(f[1].split(' ')[0]) / shape[1], int(f[1].split(' ')[1]) / shape[1],
                              int(f[1].split(' ')[2]) / shape[0], int(f[1].split(' ')[3]) / shape[0]]
                picSShape2 = [int(f[2].split(' ')[0]) / shape[1], int(f[2].split(' ')[1]) / shape[1],
                              int(f[2].split(' ')[2]) / shape[0], int(f[2].split(' ')[3]) / shape[0]]
                picSShape = [picSShape1, picSShape2]
                OArea1 = OverlapArea(picIShape[0], picSShape1)
                SArea1 = (picSShape1[1] - picSShape1[0]) * (picSShape1[3] - picSShape1[2])
                IArea1 = (picIShape[0][1] - picIShape[0][0]) * (picIShape[0][3] - picIShape[0][2])
                OArea2 = OverlapArea(picIShape[0], picSShape1)
                SArea2 = (picSShape2[1] - picSShape2[0]) * (picSShape2[3] - picSShape2[2])
                IArea2 = (picIShape[1][1] - picIShape[1][0]) * (picIShape[1][3] - picIShape[1][2])
                if Overlap(picSShape1,picIShape[0]) == 1 and OArea1 / SArea1 >= th and OArea1 / IArea1 >= th and \
                   Overlap(picSShape2, picIShape[1]) == 1 and OArea2 / SArea2 >= th and OArea2 / IArea2 >= th:
                    SPGroup.append(each[:-4] + '.jpg')
        path2 = path_des + 'SelectTwo/' + str(num - 1)
        if os.path.exists(path2):
            shutil.rmtree(path2)
        os.mkdir(path2)
        for each in SPGroup:
            shutil.copyfile(pathG_pic + each, path2 + '/' + each)
