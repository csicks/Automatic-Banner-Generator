from sklearn.cluster import KMeans,MeanShift,SpectralClustering,AffinityPropagation
import shutil
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import csv
import json

def GetMin(n, c, shape):
    a=min(abs(n[0]-c[0]),abs(n[0]-c[1]),abs(n[1]-c[0]),abs(n[1]-c[1]))
    b=min(abs(n[2]-c[2]),abs(n[2]-c[3]),abs(n[3]-c[2]),abs(n[3]-c[3]))
    return min(a/shape[0], b/shape[1])

def Overlap(n, c):
    if c[0] < n[0] < c[1] and c[2] < n[2] < c[3]:
        return 1
    elif c[0] < n[0] < c[1] and c[2] < n[3] < c[3]:
        return 1
    elif c[0] < n[1] < c[1] and c[2] < n[2] < c[3]:
        return 1
    elif c[0] < n[1] < c[1] and c[2] < n[3] < c[3]:
        return 1
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

            th = 1 / 60
            cluster = []
            if len(fourCorner) > 0:
                k = 0
                for each in fourCorner:
                    state = 0
                    if k == 0:
                        cluster.append([each])
                    else:
                        isInd = 1
                        DD=[]
                        for aa in cluster:
                            distance=[]
                            for ii in aa:
                                if Overlap(each,ii) == 1:
                                    cluster.remove(aa)
                                    aa.append(each)
                                    cluster.append(aa)
                                    state = 1
                                    break
                                distance.append(GetMin(each,ii,shape))
                            if state == 1:
                                break
                            minD=min(distance)
                            if minD < th:
                                isInd = 0
                                DD.append([minD,cluster.index(aa)])
                        if state == 1:
                            k += 1
                            continue
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
                            In=cluster[num]
                            cluster.remove(In)
                            In.append(each)
                            cluster.append(In)
                    k += 1

            collection=[]
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

            path="G:/浙大实习/text聚类/"
            fobj = open(path + name[0:len(row[0])-4] + ".txt", 'w')
            for each in collection:
                fobj.write('\n' + str(each[0]) + ' ' + str(each[1]) + ' ' + str(each[2]) + ' ' + str(each[3]))
            fobj.close()
            # print(len(collection))









