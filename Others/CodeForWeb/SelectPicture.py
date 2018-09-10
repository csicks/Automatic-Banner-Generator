import csv
import json
from operator import itemgetter
import cv2
import numpy as np

fixheight = 50
roinum = 0  # 主要轮廓的个数
roiC = 0  # 形状指数
threshold_C = 0.4  # 形状指数差异阈值，越小越严格，在ROI数小于输入图的ROI数时使用
threshold_ar = 0.4  # 长宽比差异阈值，越小越严格
threshold_match = 0.4  # 异或后的相似度差异阈值,越小越严格


# 比较两幅图像，返回匹配差异值
def compare(ori, dst, fixheight, scaledwidth, aspect_ratio):
    dheight, dwidth = dst.shape
    dscaledwidth = int(fixheight * dwidth * 1.0 / dheight)

    if dscaledwidth == 0:
        return 100
    dst = cv2.resize(dst, (dscaledwidth, fixheight))
    if dscaledwidth < scaledwidth:
        # 缩放后宽度小于输入图的宽度，在两边填充零
        beg = int((scaledwidth - dscaledwidth) * 1.0 / 2)
        fore = np.zeros((fixheight, beg), 'int')
        tail = np.zeros((fixheight, scaledwidth - dscaledwidth - beg), 'int')
        temp = np.hstack((fore, dst))
        dstt = np.hstack((temp, tail))
    else:
        dstt = cv2.resize(dst, (scaledwidth, fixheight))

    ret, dst_gray = cv2.threshold(dst, 253, 255, cv2.THRESH_BINARY)
    m_roinum, m_ar, C = findInfo(dst_gray)

    # mask图的主轮廓长宽比差异较大
    if abs(aspect_ratio - m_ar) * 1.0 / aspect_ratio > threshold_ar:
        return 100
    # mask图的主轮廓数量大于输入图主轮廓数
    if m_roinum > roinum:
        return 100
    # mask图的主轮廓数量小于等于输入图主轮廓数，用形状指数筛一次
    if (abs(roiC - C) * 1.0 / roiC > threshold_C):
        return 100

    count = np.count_nonzero(dstt)
    if count == 0:  # 去除全黑的图像
        return 100

    dstt = dstt * 1.0 / 255
    dstt = np.array(dstt, dtype='bool')

    dstt = dstt ^ ori
    count = np.count_nonzero(dstt)

    match = count * 1.0 / dstt.size
    return match


# 复制搜索出的图片到指定目录中，仅作调试用
def ouput_result_txt(output_filepath, allimg):
    f = open(output_filepath + "result__.txt", "w")
    i = 0
    for it in allimg:
        i += 1
        f.write(it[0] + "\n")
    f.close()
    return output_filepath + 'result__.txt'


# 找主要轮廓，返回其在轮廓集中的索引，面积和周长
def findMainContours(contours):
    s_contours = []
    if len(contours) == 0:
        return s_contours
    for idx in range(len(contours)):
        s = cv2.contourArea(contours[idx])
        p = cv2.arcLength(contours[idx], True)
        s_contours.append((idx, s, p))

    s_contours.sort(key=itemgetter(1), reverse=True)

    prev_s = s_contours[0][1]
    for it in range(1, len(s_contours)):
        if s_contours[it][1] * 1.0 / prev_s < 0.2:
            del s_contours[it:]
            break

    return s_contours


# 求形状指数
def findC(s_contours):
    ssum = 0
    psum = 0
    for it in s_contours:
        ssum += it[1]
        psum += it[2]
    if ssum == 0:
        return 0
    C = psum * psum * 1.0 / ssum
    return C


# 找出输入图片(二值化图像)的主要轮廓，返回有几个主要轮廓，以及总的长宽比,形状指数
def findInfo(ori_mask):
    image, contours, hierarchy = cv2.findContours(ori_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找主轮廓
    s_contours = findMainContours(contours)
    global roinum
    roinum = len(s_contours)
    if roinum < 1:
        return roinum, 0, 0
    if roinum > 1:
        xl = []
        yl = []
        for it in s_contours:
            x, y, w, h = cv2.boundingRect(contours[it[0]])
            xl.append(x)
            xl.append(x + w)
            yl.append(y)
            yl.append(y + h)
        w = max(xl) - min(xl)
        h = max(yl) - min(yl)
    else:
        x, y, w, h = cv2.boundingRect(contours[s_contours[0][0]])

    C = findC(s_contours)  # 求形状指数
    aspect_ratio = h * 1.0 / w
    return roinum, aspect_ratio, C


def productFilter(data_filepath, boundary_filepath, input_filepath, output_filepath):
    # boundary_filepath='G:/Font/image_boundary/'
    # input_filepath='G:/Font/1.png'
    # output_filepath='G:/Font/'
    # csvFile = open("G:/浙大实习/数据label/1-8238.csv", "r")
    csvFile = open(data_filepath, "r")
    reader = csv.reader(csvFile)

    img = cv2.imread(input_filepath)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 5)  # 中值滤波去噪
    ret, ori_mask = cv2.threshold(img_gray, 253, 255, cv2.THRESH_BINARY_INV)

    global roinum, roiC
    roinum, aspect_ratio, roiC = findInfo(ori_mask)

    allimg = []

    oheight, owidth = img_gray.shape
    scaledwidth = int((fixheight * 1.0) / oheight * owidth)
    ori = cv2.resize(ori_mask, (scaledwidth, fixheight))
    ori = ori / 255
    ori_bin = np.array(ori, dtype='bool')
    count_ar = 0

    for item in reader:
        info = json.loads(item[1])
        match = 100
        maskname = None
        for item in info["products"]:
            width = item['width']
            height = item['height']
            mask = item['mask']  # 产品的边缘检测图（前8000+数据含有，后5000+数据没有这个字段）

            # 计算初步长宽比的差异
            match_ar = abs(aspect_ratio - height * 1.0 / width) * 1.0 / aspect_ratio

            # 选取小于阈值的图像进行下一步比对
            if match_ar < threshold_ar:
                count_ar += 1
                img_data = cv2.imread(boundary_filepath + mask, cv2.IMREAD_GRAYSCALE)
                temp = compare(ori_bin, img_data, fixheight, scaledwidth, aspect_ratio)
                # 若一幅图有多个mask，选取其中差异值最小的那个
                if temp < match:
                    match = temp
                    maskname = mask
        if match < threshold_match:
            allimg.append((maskname, match))
    allimg = sorted(allimg, key=itemgetter(1))  # 按差异值从小到大排序
    return ouput_result_txt(output_filepath, allimg)


def FindData(x, collection):
    for each in collection:
        if x == each:
            return 1
    return 0


def prob(x, k, mu, sigma):
    index = -0.5 * pow((x - mu) / sigma, 2)
    numerator = k * pow(np.e, index)
    denominator = sigma * pow(2 * np.pi, 0.5)
    return numerator / denominator


def SelectPicture(type0, path_In_CSV, path_In_Boundary, path_In_Product, path_Out_ProductMatch, path_In_Result,
                  path_Out):
    path1 = productFilter(path_In_CSV, path_In_Boundary, path_In_Product, path_Out_ProductMatch)
    f1 = open(path1, 'r')
    k = 0
    nameGroup = []
    for line in f1:
        if k != 0:
            if line[len(line) - 1] == '\n':
                line = line[:-1]
            string = line.split('_')[0]
            nameGroup.append(string + '.jpg')
        k += 1
    f1.close()

    SPGroup = []

    with open(path_In_CSV) as fr:
        rows = csv.reader(fr)
        for row in rows:
            if FindData(row[0], nameGroup) == 1:
                item = json.loads(row[1])
                if item['productsNum'] == 1:
                    SPGroup.append(row[0])

    if type0 == 1:
        path3 = path_In_Result
        f3 = open(path3, 'r')
        k = 0
        SPGroupF = []
        ClassGroup = []
        count = 0
        for line in f3:
            if k != 0:
                if line[len(line) - 1] == '\n':
                    line = line[:-1]
                string = line.split(' ')[0]
                if FindData(string, SPGroup) == 1:
                    if line.split(' ')[1] not in ClassGroup:
                        ClassGroup.append(line.split(' ')[1])
                    SPGroupF.append([string, line.split(' ')[1]])
                    count += 1
            k += 1
        f3.close()

        path2 = path_Out
        f2 = open(path2, 'w')
        f2.write(str(count) + ' ' + str(len(ClassGroup)))
        for each in SPGroupF:
            f2.write('\n' + each[0] + ' ' + each[1])
        f2.close()

    elif type0 == 2:
        path3 = path_In_Result
        f3 = open(path3, 'r')
        k = 0
        SPGroupF = []
        ClassGroup = []
        count = 0
        for line in f3:
            if k != 0:
                if line[len(line) - 1] == '\n':
                    line = line[:-1]
                string = line.split(' ')[0]
                if FindData(string, SPGroup) == 1:
                    if line.split(' ')[1] not in ClassGroup:
                        ClassGroup.append(line.split(' ')[1])
                    SPGroupF.append([string, line.split(' ')[1]])
                    count += 1
            k += 1
        f3.close()

        path2 = path_Out
        output = []
        f2 = open(path2, 'w')
        f2.write(str(count) + ' ' + str(len(ClassGroup)))
        for each in SPGroupF:
            f2.write('\n' + each[0] + ' ' + each[1])
            output.append([each[0], each[1]])
        f2.close()
        return output


type0 = 2
path_In_CSV = "G:/浙大实习/数据label/JsonDataForBanner.csv"
path_In_Boundary = 'G:/Font/image_boundary/'
path_In_Product = 'G:/Font/1.png'
path_Out_ProductMatch = 'G:/Font/'
# path_In_Result="G:/浙大实习/text_product聚类/resultOne2OneAngle.txt"
path_In_Result = "G:/浙大实习/text_product聚类/resultOne2TwoAngleFinal.txt"
# path_Out="G:/浙大实习/text_product聚类/SelectedPictureOne.txt"
path_Out = "G:/浙大实习/text_product聚类/SelectedPictureTwo.txt"
output = SelectPicture(type0, path_In_CSV, path_In_Boundary, path_In_Product, path_Out_ProductMatch, path_In_Result,
                       path_Out)
x = 1
