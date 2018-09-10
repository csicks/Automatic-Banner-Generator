import sys
import os, os.path, shutil

type0 = "oneNew"
type1 = 1
path_pic = "G:/浙大实习/聚类结果/"

if type0 == "one":
    path1 = ""
    path2 = ""
    if type1 == 0:
        path1 = "G:/浙大实习/text_product聚类/resultRelativePosition.txt"
        path2 = "G:/浙大实习/text_product聚类/RelativePositionOne/"
    elif type1 == 1:
        path1 = "G:/浙大实习/text_product聚类/resultOne2OneAngle.txt"
        path2 = "G:/浙大实习/text_product聚类/AngleOne/"
    if os.path.exists(path2):
        shutil.rmtree(path2)
    os.mkdir(path2)
    k = 0
    count = 0
    for line in open(path1, 'r'):
        if k == 0:
            count = int(line[:-1])
        elif k != 0:
            if k != count:
                line = line[:-1]
            sg = line.split(' ')
            if not os.path.exists(path2 + sg[1]):
                os.mkdir(path2 + sg[1])
            shutil.copyfile(path_pic + sg[0], path2 + sg[1] + '/' + sg[0])
        k += 1

elif type0 == "two":
    path0 = "G:/浙大实习/text_product聚类/resultOne2TwoAngleFinal.txt"
    path1 = "G:/浙大实习/text_product聚类/resultOne2TwoAngleBoth.txt"
    path2 = "G:/浙大实习/text_product聚类/AngleTwo/"
    if os.path.exists(path2):
        shutil.rmtree(path2)
    os.mkdir(path2)
    k = 0
    count = 0
    collection = []
    for line in open(path0, 'r'):
        if not os.path.exists(path2 + line[:-1]):
            os.mkdir(path2 + line[:-1])
        collection.append(line[:-1])
    for line in open(path1, 'r'):
        if k == 0:
            count = int(line[:-1])
        else:
            if k != count:
                line = line[:-1]
            sg = line.split(' ')
            if sg[1] + sg[2] == collection[0]:
                shutil.copyfile(path_pic + sg[0], path2 + collection[0] + '/' + sg[0])
            elif sg[1] + sg[2] == collection[1]:
                shutil.copyfile(path_pic + sg[0], path2 + collection[1] + '/' + sg[0])
            elif sg[1] + sg[2] == collection[2]:
                shutil.copyfile(path_pic + sg[0], path2 + collection[2] + '/' + sg[0])
            elif sg[1] + sg[2] == collection[3]:
                shutil.copyfile(path_pic + sg[0], path2 + collection[3] + '/' + sg[0])
        k += 1

elif type0 == "oneNew":
    path1 = "G:/浙大实习/text_product聚类/SelectedPictureOne.txt"
    path2 = "G:/浙大实习/text_product聚类/NewAngleOne2One/"
    if os.path.exists(path2):
        shutil.rmtree(path2)
    os.mkdir(path2)
    k = 0
    for line in open(path1, 'r'):
        if k != 0:
            if line[len(line) - 1] == '\n':
                line = line[:-1]
            sg = line.split(' ')
            if not os.path.exists(path2 + sg[1]):
                os.mkdir(path2 + sg[1])
            if os.path.exists(path_pic + sg[0]):
                shutil.copyfile(path_pic + sg[0], path2 + sg[1] + '/' + sg[0])
        k += 1

elif type0 == "twoNew":
    path1 = "G:/浙大实习/text_product聚类/SelectedPictureTwo.txt"
    path01 = "G:/浙大实习/text_product聚类/resultOne2TwoAngle1.txt"
    path02 = "G:/浙大实习/text_product聚类/resultOne2TwoAngle2.txt"
    path2 = "G:/浙大实习/text_product聚类/NewAngleOne2Two/"
    path3 = "G:/浙大实习/text_product聚类/NewAngleOne2Two/Class1/"
    path4 = "G:/浙大实习/text_product聚类/NewAngleOne2Two/Class2/"
    if os.path.exists(path2):
        shutil.rmtree(path2)
    os.mkdir(path2)
    os.mkdir(path3)
    os.mkdir(path4)
    k = 0
    for line in open(path1, 'r'):
        if k != 0:
            if line[len(line) - 1] == '\n':
                line = line[:-1]
            sg = line.split(' ')
            if not os.path.exists(path2 + sg[1]):
                os.mkdir(path2 + sg[1])
            shutil.copyfile(path_pic + sg[0], path2 + sg[1] + '/' + sg[0])
        k += 1
    k = 0
    for line in open(path01, 'r'):
        if k != 0:
            if line[len(line) - 1] == '\n':
                line = line[:-1]
            sg = line.split(' ')
            if not os.path.exists(path3 + sg[1]):
                os.mkdir(path3 + sg[1])
            shutil.copyfile(path_pic + sg[0], path3 + sg[1] + '/' + sg[0])
        k += 1
    k = 0
    for line in open(path02, 'r'):
        if k != 0:
            if line[len(line) - 1] == '\n':
                line = line[:-1]
            sg = line.split(' ')
            if not os.path.exists(path4 + sg[1]):
                os.mkdir(path4 + sg[1])
            shutil.copyfile(path_pic + sg[0], path4 + sg[1] + '/' + sg[0])
        k += 1
