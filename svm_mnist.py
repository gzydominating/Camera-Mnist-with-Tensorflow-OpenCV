#import os
#import sys
import numpy as np
#import time
from sklearn import svm
#from sklearn.externals import joblib
#from PIL import Image
import tensorflow.examples.tutorials.mnist.input_data as input_data

'''
def get_file_list(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png")]
	

def get_img_name_str(imgpath):
    return imgpath.split(os.path.sep)[-1]


def img2vector(imgfile):
    #print("in img2vector func--para:{}".format(imgFile))
    img = Image.open(imgfile).convert('L')
    img_arr = np.array(img, 'i') # 20px * 20px 灰度图像
    img_normalization = np.round(img_arr/255) # 对灰度值进行归一化
    img_arr2 = np.reshape(img_normalization, (1,-1)) # 1 * 400 矩阵
    return img_arr2
    
	
def read_and_convert(imgFileList):
    dataLabel = [] # 存放类标签
    dataNum = len(imgFileList)
    dataMat = np.zeros((dataNum, 400)) # dataNum * 400 的矩阵
    for i in range(dataNum):
        imgNameStr = imgFileList[i]
        imgName = get_img_name_str(imgNameStr)  # 得到 数字_实例编号.png
        #print("imgName: {}".format(imgName))
        classTag = imgName.split(".")[0].split("_")[0] # 得到 类标签(数字)
        #print("classTag: {}".format(classTag))
        dataLabel.append(classTag)
        dataMat[i,:] = img2vector(imgNameStr)
    return dataMat, dataLabel
	
	
# 读取训练数据
def read_all_data():
    cName = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    train_data_path = "mnist-image\\train\\0"
    flist = get_file_list(train_data_path)
    dataMat, dataLabel = read_and_convert(flist)
    for c in cName:
        train_data_path_ = "mnist-image\\train\\" + c
        flist_ = get_file_list(train_data_path_)
        dataMat_, dataLabel_ = read_and_convert(flist_)
        dataMat = np.concatenate((dataMat, dataMat_), axis=0)
        dataLabel = np.concatenate((dataLabel, dataLabel_), axis=0)
    #print(dataMat.shape)
    #print(len(dataLabel))
    return dataMat, dataLabel
'''
#dataMat, dataLabel = read_all_data()
mnist = input_data.read_data_sets("MNIST_data/")
dataMat = mnist.train.images
dataLabel = mnist.train.labels
#print(dataMat)
print(dataLabel)
#print(np.argmax(dataLabel[0]))


print('build model...')
clf = svm.SVC(decision_function_shape='ovr')
clf.fit(dataMat, dataLabel)
joblib.dump(clf, "saved_model.m")
print('done')

# 对10个数字进行分类测试
'''
def test():
    tbasePath = "mnist-image\\test\\"
    tcName = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    tst = time.clock()
    allErrCount = 0
    allErrorRate = 0.0
    allScore = 0.0
    for tcn  in tcName:
        testPath = "mnist-image\\test\\" + tcn
        #print("class " + tcn + " path is: {}.".format(testPath))
        tflist = get_file_list(testPath)
        #tflist
        tdataMat, tdataLabel = read_and_convert(tflist)
        print("test dataMat shape: {0}, test dataLabel len: {1} ".format(tdataMat.shape, len(tdataLabel)))

        #print("test dataLabel: {}".format(len(tdataLabel)))
        pre_st = time.clock()
        preResult = clf.predict(tdataMat)
        pre_et = time.clock()
        print("Recognition  " + tcn + " spent {:.4f}s.".format((pre_et-pre_st)))
        #print("predict result: {}".format(len(preResult)))
        errCount = len([x for x in preResult if x!=tcn])
        print("errorCount: {}.".format(errCount))
        allErrCount += errCount
        score_st = time.clock()
        score = clf.score(tdataMat, tdataLabel)
        score_et = time.clock()
        print("computing score spent {:.6f}s.".format(score_et-score_st))
        allScore += score
        print("score: {:.6f}.".format(score))
        print("error rate is {:.6f}.".format((1-score)))
        print("---------------------------------------------------------")


    tet = time.clock()
    print("Testing All class total spent {:.6f}s.".format(tet-tst))
    print("All error Count is: {}.".format(allErrCount))
    avgAccuracy = allScore/10.0
    print("Average accuracy is: {:.6f}.".format(avgAccuracy))
    #print("Average error rate is: {:.6f}.".format(1-avgScore))
'''	

# 摄像头识别数字

import cv2

cap = cv2.VideoCapture(0)
while(1):
    ret, frame = cap.read()
    cv2.rectangle(frame,(270,200),(340,270),(0,0,255),2)
    cv2.imshow("capture", frame)
    roiImg = frame[200:270,270:340]
    img = cv2.resize(roiImg,(20,20))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_normalization = np.round(img/255) # 对灰度值进行归一化
    img_arr2 = np.reshape(img_normalization, (1,-1)) # 1 * 400 矩阵
	
    #clf = joblib.load("saved_model.m")
    preResult = clf.predict(img_arr2)

    #predicts=predictions.tolist() 
    label=preResult[0]
    #result=label.index(max(label))
    print('result num:')
    print(label)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

