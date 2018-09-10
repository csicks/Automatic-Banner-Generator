from sklearn.cluster import KMeans,MeanShift,SpectralClustering
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
    for row in rows:
        img = Image.open("G:/浙大实习/allImages-8337/" + row[0])
        shape = img.size
        ImageShape.append(shape)
        name = row[0]

        item = json.loads(row[1])

        if item['productsNum'] == 1:

            tProduct = item['products'][0]['y']
            bProduct = item['products'][0]['y']+item['products'][0]['height']
            lProduct = item['products'][0]['x']
            rProduct = item['products'][0]['x']+item['products'][0]['width']
            cornerProduct = (tProduct,bProduct,lProduct,rProduct)

            centersPercent = []
            fourCorner = []
            for each in item["texts"]:
                if each['label'] != "logo":
                    x = each['x']
                    y = each['y']
                    width = each['width']
                    height = each['height']
                    if height < 20 or width < 20:
                        continue
                    fourCorner.append((y,y+height,x,x+width))
                    centersPercent.append((y / shape[1], (y + height) / shape[1], x / shape[0], (x + width) / shape[0]))

            if len(centersPercent)>0:
                cex = []
                cey = []
                for po in centersPercent:
                    cex.append((po[2]+po[3])/2)
                    cey.append((po[0]+po[1])/2)
                X = np.array([cex,cey])
                X = X.T
                y_pred = MeanShift(bandwidth=0.4).fit_predict(X)

            tCollection = []
            bCollection = []
            lCollection = []
            rCollection = []
            if len(set(y_pred)) == 1 and len(fourCorner)>0:
                for corner in fourCorner:
                    tCollection.append(corner[0])
                    bCollection.append(corner[1])
                    lCollection.append(corner[2])
                    rCollection.append(corner[3])
                tText = min(tCollection)
                bText = max(bCollection)
                lText = min(lCollection)
                rText = max(rCollection)
                cornerText = (tText,bText,lText,rText)

                cornerAbsolution = (cornerText[0]/shape[1],cornerText[2]/shape[0],(cornerText[0]-cornerProduct[0])/shape[1],(cornerText[1]-cornerProduct[1])/shape[1],(cornerText[2]-cornerProduct[2])/shape[0],(cornerText[3]-cornerProduct[3])/shape[0])
                cornerAbsolutionCollection.append(cornerAbsolution)
                # print(cornerAbsolution)
                fileName.append(name)



    py = []
    px = []
    tPo = []
    bPo = []
    lPo = []
    rPo = []
    for po in cornerAbsolutionCollection:
        py.append(po[0])
        px.append(po[1])
        tPo.append(po[2])
        bPo.append(po[3])
        lPo.append(po[4])
        rPo.append(po[5])
    X = np.array([py,px,tPo, bPo,lPo,rPo])
    X = X.T

    # y_pred = KMeans(n_clusters=6,random_state=9).fit_predict(X)
    # y_pred = MeanShift(bin_seeding=0.4).fit_predict(X)
    y_pred = SpectralClustering(n_clusters=5,random_state=9).fit_predict(X)
    print(set(y_pred))
    for index,label in enumerate(y_pred):
        # if label==5:
            # print(fileName[index],X[index,:])
        print(index,y_pred[index])
        # if os.path.exists("result/"+str(label))==0:
        #     os.makedirs("result/"+str(label))
        # shutil.copy("../Json/DrawLabelImgRe/"+fileName[index], "result/"+str(label)+"/"+fileName[index])