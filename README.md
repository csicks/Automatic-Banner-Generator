# Automatical-Banner-Generator
Algorithms of a automatical banner generator (Python)

This is an implementation of a automatical banner generator, which receives a background picture, a model picture and several texts as input and then it outputs a well-designed banner. The algorithms are data-driven algorithms, relying on a dataset which contains over 8000 banners designed by human desigeners. If you need the dataset, please contact me by email yxchen11@outlook.com. Or you can create a dataset by yourself.

These codes are part of my internship work in inlab, Zhejiang University. We tried to build a website by Django to realize the automatical banner generator and my part was designind and implement the generating algorithms.

There are few comments in these codes. But two slides are included in this repository, one in Chinese and another one in English. Though they are not made to explain the algorithms, they may help you understand these algorithms.

In our algorithms, we only consider such situation: one model, one or two text groups.

Please note that these codes are only tested under Windows. Therefore, if you would like to run these codes under Linux or other OS, you may need to modify these codes by yourself.


## Run the Codes:

Before running the codes, you need to modify all the file path in the codes.

After that,

Firstly, you need to run the file 'ClusteringTextMerge.py'  and then you will get a folder filled with txt file in which there are positions of all the  text groups in a given picture.

Secondly, you need to run the file 'ClusteringAllOne2OneAngle.py' or 'ClusteringAllOne2TwoAngle.py'. 'One2One' means there are one model and one text group. 'One2Two' means there are one model and two text groups. After running the codes, you will get several txt files in which you will get some information about the clustering results.

Thirdly, you need to run the file 'productFilter.py' which will select the pictures having similar model shape with the input model. And the output is a txt file stating the pictures' name.

Fourthly, you need to run the file 'SelectPicture.py' which will select some pictures according to the file generated in the third step.

Fifthly, you need to run the file 'GaussianClusterOne2One.py' or 'GaussianClusterOne2Two.py'. The meaning of 'One2One' and 'One2Two' has been stated in the second step. After running this file, you will get the abstract layout of each clustered class.

Finally, you need to run the file 'InsideTextLayoutOneFixed.py' or 'InsideTextLayoutTwoFixed.py' which will decide the layouts in the text groups. And you will get the final results. Please note that you need to put the font files to the right path and change the names of the files so that you will get no error for this part.


## Final words:

As is said above, here are only part of the project, mainly the algorithms with no UI.

I have also attahed some other codes of this project. Some of them are used to see some process pictures or results. Some of them are codes for convenience. And some of them are algorithms codes modified to function version which are more easy to use in the websites. I put them here in case that you may need them.

There are many codes and features which can be improved but I do not plan to do that.

Finished in 2018.
