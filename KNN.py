# -*- coding: utf-8 -*-     支持文件中出现中文字符
#########################################################################
"""
Created on 2022.11.12
@author: Zoey
@version: Python 3.10
@简写说明
  + freq: frequently
  + calc: calculate
  + Vec: Vector
  +
  +
  +

"""
#########################################################################
import Base #第一次作业, 基本数据处理及特征提取
from LSM import xlsRead #第二次作业, 读取数据
from DecisionTree import * #第三次作业
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler #归一化数据

def RoughKNN(knownData:np.ndarray, testData:np.ndarray, k:int) -> np.ndarray:
    '''	@func: 暴力求解KNN
        @para  knownData: 已知数据集，含标签，浮点数据
               testData: 待测试数据集，不含标签，浮点数据
               k: KNN参数
        @return: 预测的标签列表, 应与testData长度相同
    '''
    Prelabel = np.zeros(len(testData),dtype=knownData[-1].dtype) #预测输出的列表
    distance = np.zeros(len(knownData)) #已知点距离待测点的距离
    L = np.array(k,dtype=knownData.dtype) #存放列表
    for i in range(len(testData)): #遍历每个待测样本
        for j in range(len(knownData)):
            distance[j] = np.sqrt(np.sum( (testData[i]-knownData[j][:-1])**2 ))
        indexOUT = np.argsort(distance) #取排序后的索引值
        L = indexOUT[0:k] #距离最小的前k个数据对应的索引值
        Prelabel[i] = MostFreq(knownData[:,-1][L].copy()) #获取标签中出现次数最多的标签
    return Prelabel

def MostFreq(Vec:np.ndarray):
    '''	@func: 获取某个向量(行向量)中出现次数最多的元素
        @para  Vec:传入的行向量
        @return: 出现次数最多的元素
    '''
    unique_value,occurrence = np.unique(Vec, return_counts=True)
    Dict = dict(zip(unique_value,occurrence))
    sortedDict = sorted(Dict.items(), key=lambda d:d[1]) # type: ignore #按value排序,升序
    return sortedDict[-1][0] #返回最后一个, 样本数最多的元素


# USE_MY_ALGORITHM = 1 #决定是否调库实现
USE_MY_ALGORITHM = 0
SMALL_DATA = 1 #小数据集，直接暴力KNN
# SMALL_DATA = 0 #使用KD树进行搜索


# 主函数
def main():
    print("Please Wait for a few seconds...\n")
    K = 5 #KNN中的K参数
    for e in range(113,115):
        # 1. 数据预处理, 得到标签
        data = Base.getdata(f"{__file__}/../data/20151026_{e}", 0) #读取第一列的数据
        data_f = Base.butter_band_filter(data) #滤波
        num = 150 #划分的子列表数
        s = [data_f[i:i+num] for i in range(0, len(data_f), num)] #列表划分
        labels = labelGet2(s) #获取标签

        # 2. 读取特征表, 并归一化，再加上标签
        _, featTable = xlsRead(f"{__file__}/../data/data_{e}.xls", "SharpWave") #读取属性列表
        norm = MinMaxScaler(feature_range=(0,1))  #归一化
        featTable_norm = norm.fit_transform(featTable)
        dataset = np.hstack((featTable_norm,np.array([labels]).T)) #注意维度关系
        
        # 3. 随机划分训练测试集
        train = 0.8; test = 1-train #训练与测试集的分配
        # 方式一
        # index = np.zeros(len(dataset),dtype=int) #或者直接生成bool数组
        # index[0:int(train*len(dataset))] = 1
        # np.random.shuffle(index)
        # trainSet = dataset[index.astype(bool)] #注意这里要用bool类型的，否则花式索引会出错
        # testSet = dataset[(1-index).astype(bool)]
        # 方式二
        index = np.arange(len(dataset),dtype=int)
        np.random.shuffle(index) #打乱列表
        trainSet = dataset[index[:int(train*len(dataset))]]
        testSet = dataset[index[int(train*len(dataset)):]]

        # 4. 调库实现
        if(not USE_MY_ALGORITHM):
            nbrs = NearestNeighbors(n_neighbors=K,algorithm='kd_tree').fit(trainSet[:,:-1]) #不含标签列
            distances, indices = nbrs.kneighbors(testSet[:,:-1]) #拿测试集去试
            labelOUT = [MostFreq(item) for item in trainSet[:,-1][indices]]
            print("Predicted LabelList:\n",labelOUT)
            print("Source LabelList:\n",testSet[:,-1])
            print("Accuracy: ",ACC_Calc(testSet[:,-1], labelOUT),'\n')

        # 4. 暴力KNN
        elif(SMALL_DATA):
            labelOUT = RoughKNN(trainSet, testSet[:,:-1],K)
            print("Predicted LabelList:\n",labelOUT)
            print("Source LabelList:\n",testSet[:,-1])
            print("Accuracy: ",ACC_Calc(testSet[:,-1], labelOUT),'\n')
        
        # 5. 自建KD树搜索——还没做。。。
        else:
            pass


if __name__ == "__main__":
    main()
    input("按下任意键继续...")
