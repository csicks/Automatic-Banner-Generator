import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os, os.path, shutil
from sklearn.neighbors import kde
from PIL import Image
import seaborn as sns
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D


def prob(x, k, mu, sigma):
    index = -0.5 * pow((x - mu) / sigma, 2)
    numerator = k * pow(np.e, index)
    denominator = sigma * pow(2 * np.pi, 0.5)
    return numerator / denominator


def ReadData():
    pathSource = "G:/浙大实习/text_product聚类/SelectedPictureOne.txt"
    pathText = "G:/浙大实习/text聚类/"
    pathProduct = "G:/浙大实习/text_product聚类/productCorner.txt"
    pathPicture = "G:/浙大实习/聚类结果/"

    picName = []
    classNum = int(open(pathSource, 'r').readline()[:-1].split(' ')[1])
    classCnt = 0

    while classCnt < classNum:
        picTemp = []
        k = 0
        tNum = 0
        for line in open(pathSource, 'r'):
            if k == 0:
                tNum = int(line[:-1].split(' ')[0])
            else:
                if k != tNum:
                    line = line[:-1]
                sg = line.split(' ')
                if sg[1] == 'x' + str(classCnt):
                    picTemp.append(sg[0])
            k += 1
        if len(picTemp)!=0:
            picName.append(picTemp)
        classCnt += 1

    collection = []
    for each in picName:
        cData = []
        for item in each:
            data = [item]
            img = Image.open(pathPicture + item)
            shape = img.size
            path1 = pathText + item[:-4] + '.txt'
            k = 0
            sGroup = []
            for line in open(path1, 'r'):
                if k != 0:
                    if line[len(line) - 1] == '\n':
                        sGroup = line[:-1].split(' ')
                    else:
                        sGroup = line.split(' ')
                    break
                k += 1
            tData = []
            if os.path.getsize(path1) == 0:
                continue
            else:
                tData.append(int(sGroup[0]) / shape[1])
                tData.append(int(sGroup[1]) / shape[1])
                tData.append(int(sGroup[2]) / shape[0])
                tData.append(int(sGroup[3]) / shape[0])
            data.append(tData)
            pData = []
            k = 0
            for line in open(pathProduct, 'r'):
                if k != 0:
                    if line[len(line) - 1] == '\n':
                        line = line[:-1]
                    sg = line.split(' ')
                    if item == sg[4]:
                        pData.append(int(sg[0]) / shape[1])
                        pData.append(int(sg[1]) / shape[1])
                        pData.append(int(sg[2]) / shape[0])
                        pData.append(int(sg[3]) / shape[0])
                        break
                k += 1
            data.append(pData)
            cData.append(data)
        collection.append(cData)
    return [collection]


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
    if Overlap(r1, r2) == 1:
        y1 = max(r1[0], r2[0])
        y2 = min(r1[1], r2[1])
        x1 = max(r1[2], r2[2])
        x2 = min(r1[3], r2[3])
        return (y2 - y1) * (x2 - x1)
    else:
        return 0


def ShapeSimilarity(r1, r2):
    h1 = r1[1] - r1[0]
    w1 = r1[3] - r1[2]
    h2 = r2[1] - r2[0]
    w2 = r2[3] - r2[2]
    hr = h1 / h2 if h2 > h1 else h2 / h1
    wr = w1 / w2 if w2 > w1 else w2 / w1
    return [hr, wr]


TotalData = ReadData()
data = TotalData[0]

path = "G:/浙大实习/text_product聚类/One2OneProduct.txt"

# 产品位置聚类
ProductPosition = []
for ccData in data:
    ZPData = []
    for each in ccData:
        ZPData.append([(each[2][0] + each[2][1]) / 2, (each[2][2] + each[2][3]) / 2])
    pp = kde.KernelDensity(kernel='gaussian', bandwidth=0.3).fit(ZPData)
    inputData = []
    iData = []
    for a1 in np.linspace(0, 1, 50):
        temp = []
        for a2 in np.linspace(0, 1, 50):
            inputData.append([a1, a2])
            temp.append(a2)
        iData.append(temp)
    iData = np.array(iData)
    outputData = pp.score_samples(inputData)
    count = 0
    tenData = []
    TTData = []
    for each in outputData:
        tenData.append(pow(np.e, each))
        count += 1
        if count == 50:
            TTData.append(tenData)
            tenData = []
            count = 0
    TTData = np.array(TTData)
    yData = iData.T
    xData = iData
    position = np.argmax(TTData)
    y = position % 50 - 1
    x = int((position - (y + 1)) / 50)
    ProductPosition.append([yData[x][y], xData[x][y]])

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # surf = ax.plot_surface(xData, yData, TTData, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #
    # # Customize the z axis.
    # ax.set_zlim(0, 5)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

# 产品长宽聚类
ProductAll = []
f2 = open(path, 'w')
kP = 0
for ccData in data:
    ppData = []
    for each in ccData:
        ppData.append([each[2][1] - each[2][0], each[2][3] - each[2][2]])
    pp = kde.KernelDensity(kernel='gaussian', bandwidth=0.5).fit(ppData)
    inputData = []
    for a1 in np.linspace(0, 1, 100):
        for a2 in np.linspace(0, 1, 100):
            inputData.append([a1, a2])
    outputData = pp.score_samples(inputData)
    density = []
    for each in outputData:
        density.append(pow(np.e, each))
    index = density.index(max(density))
    pw = np.linspace(0, 1, 100)[index % 100]
    pl = np.linspace(0, 1, 100)[int((index - pw) / 100)]
    ProductAll.append([ProductPosition[kP][0], ProductPosition[kP][1], pl, pw])
    f2.write('\n' + str(ProductPosition[kP][0]) + ' ' + str(ProductPosition[kP][1]) + ' ' + str(pl) + ' ' + str(pw))
    kP += 1
f2.close()

# 图片筛选1
NewData = []
th = 0.8
cNum = 0
for ccData in data:
    cSave = ccData[0]
    yS1 = ProductAll[cNum][0] - ProductAll[cNum][2] * 0.5
    yS2 = ProductAll[cNum][0] + ProductAll[cNum][2] * 0.5
    xS1 = ProductAll[cNum][1] - ProductAll[cNum][3] * 0.5
    xS2 = ProductAll[cNum][1] + ProductAll[cNum][3] * 0.5
    NewCCData = []
    for each in ccData:
        pData = each[2]
        pY = (pData[0] + pData[1]) / 2
        pX = (pData[2] + pData[3]) / 2
        OArea = OverlapArea([yS1, yS2, xS1, xS2], pData)
        SArea = (yS2 - yS1) * (xS2 - xS1)
        PArea = (pData[1] - pData[0]) * (pData[3] - pData[2])
        if OArea / SArea >= th and OArea / PArea >= th:
            NewCCData.append(each)
        pData1 = cSave[2]
        OArea1 = OverlapArea([yS1, yS2, xS1, xS2], pData1)
        SArea1 = (yS2 - yS1) * (xS2 - xS1)
        PArea1 = (pData1[1] - pData1[0]) * (pData1[3] - pData1[2])
        if OArea / SArea + OArea / PArea > OArea1 / SArea1 + OArea1 / PArea1:
            cSave = each
    if len(NewCCData) == 0:
        NewCCData.append(cSave)
    NewData.append(NewCCData)
    cNum += 1

# 文字位置聚类
TextPosition = []
k = 0
for ttData in NewData:
    TRPData = []
    for each in ttData:
        TRPData.append([(each[1][0] + each[1][1]) / 2 - ProductPosition[k][0],
                        (each[1][2] + each[1][3]) / 2 - ProductPosition[k][1]])
    k += 1
    # TRPData=np.array(TRPData).reshape(-1,1)
    pp = kde.KernelDensity(kernel='gaussian', bandwidth=0.5).fit(TRPData)
    inputData = []
    for a1 in np.linspace(-1, 1, 100):
        for a2 in np.linspace(-1, 1, 100):
            inputData.append([a1, a2])
    outputData = pp.score_samples(inputData)
    density = []
    for each in outputData:
        density.append(pow(np.e, each))
    index = density.index(max(density))
    xR = np.linspace(-1, 1, 100)[index % 100]
    yR = np.linspace(-1, 1, 100)[int((index - xR) / 100)]
    TextPosition.append([yR, xR])

# 文字大小聚类
path = "G:/浙大实习/text_product聚类/One2OneText.txt"
f3 = open(path, 'w')
kT = 0
for ttData in NewData:
    tpData = []
    for each in ttData:
        tpData.append([each[1][1] - each[1][0], each[1][3] - each[1][2]])
    pp = kde.KernelDensity(kernel='gaussian', bandwidth=0.5).fit(tpData)
    inputData = []
    for a1 in np.linspace(0, 1, 100):
        for a2 in np.linspace(0, 1, 100):
            inputData.append([a1, a2])
    outputData = pp.score_samples(inputData)
    density = []
    for each in outputData:
        density.append(pow(np.e, each))
    index = density.index(max(density))
    tw = np.linspace(0, 1, 100)[index % 100]
    tl = np.linspace(0, 1, 100)[int((index - tw) / 100)]
    f3.write('\n' + str(TextPosition[kT][0]) + ' ' + str(TextPosition[kT][1]) + ' ' + str(tl) + ' ' + str(tw))
    kT += 1
f3.close()
