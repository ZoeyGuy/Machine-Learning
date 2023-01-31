# -*- coding: utf-8 -*-     支持文件中出现中文字符
#########################################################################
"""
Created on 2022.11.12
@author: Zoey
@version: Python 3.10
@简写说明
  + freq: frequently
  + calc: calculate
  + feat: feature

"""
#########################################################################
import Base #第一次作业, 基本数据处理及特征提取
from LSM import xlsRead #第二次作业, 读取数据
from DecisionTree import * #第三次作业
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler #归一化数据
from sklearn import metrics
import numpy as np
import random
import pandas as pd


def calcDis(dataSet, centroids, k:int):
	'''	@func: 计算数据集样本数据离质心的欧拉距离
		@para  	dataSet: 数据集
				centroids: 质心
				k: 分类数
		@return: n*k的矩阵,即n个样本数据离k个质心的距离
	'''
	clalist=[]
	for data in dataSet:
		tmp = np.tile(data, (k, 1))  #对每一个沿x轴复制一遍，沿y轴复制k遍，方便广播
		diff = tmp - centroids  #广播相减
		squaredDiff = diff ** 2     #平方
		squaredDist = np.sum(squaredDiff, axis=1)   #和  (axis=1表示行)
		distance = squaredDist ** 0.5  #开根号
		clalist.append(distance) 
	clalist = np.array(clalist)  #返回一个每个点到质点的距离len(dateSet)*k的数组
	return clalist


def classify(dataSet, centroids, k):
	'''	@func: 根据得到的质心将原数据集进行分簇，然后计算出新的质心
		@para  	dataSet: 数据集
				centroids: 原质心列表(是一个k行m列的矩阵, 其中k代表k个质心, m行代表有m个特征, 即一条数据)
				k: 分类数
		@return: 新的质心列表及其与原质心的差别
	'''
	# 计算样本到质心的距离
	clalist = calcDis(dataSet, centroids, k)
	# 分组并计算新的质心
	minDistIndices = np.argmin(clalist, axis=1)  #axis=1 表示求出每行的最小值的下标，得到是一个列向量
	newCentroids = pd.DataFrame(dataSet).groupby(minDistIndices).mean() 
	#DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
	newCentroids = newCentroids.values
 
	# 计算变化量
	changed = newCentroids - centroids
	return changed, newCentroids


def kmeans(dataSet, k):
	'''	@func: 使用k-means进行聚类
		@para	dataSet:待聚类数据
				k:分类数
		@return: 质心数据和原数据集分类标签
	'''
	# 随机取质心
	centroids = random.sample(list(dataSet), k) #似乎不支持ndarray，要转换成list
	# 更新质心 直到变化量全为0
	changed, newCentroids = classify(dataSet, centroids, k)
	while np.any(changed != 0):
		changed, newCentroids = classify(dataSet, newCentroids, k)
 
	centroids = sorted(newCentroids.tolist())   #tolist()将矩阵转换成列表 sorted()排序
 
	# 根据质心计算每个集群
	cluster = []
	clalist = calcDis(dataSet, centroids, k) #调用欧拉距离
	# print(clalist)
	minDistIndices = np.argmin(clalist, axis=1) #表示求出每行的最小值的下标，此即对各个样本的分类结果
	for i in range(k):
		cluster.append([]) #先添加k个子列表
	for i, j in enumerate(minDistIndices):  #enumerate()可同时遍历索引和遍历元素
		cluster[j].append(dataSet[i])
		
	return minDistIndices, centroids, cluster


USE_MY_ALGORITHM = 1 #决定是否使用自己写的算法
# USE_MY_ALGORITHM = 0


def main():
	print("Please Wait for a few seconds...\n")
	np.set_printoptions(linewidth=100) #设置打印一行的长度
	N = 3 #分类数
	for e in range(113,115):
		# 1. 数据预处理, 得到标签
		data = Base.getdata(f"{__file__}/../data/20151026_{e}", 0) #读取第一列的数据
		data_f = Base.butter_band_filter(data) #滤波
		num = 150 #划分的子列表数
		s = [data_f[i:i+num] for i in range(0, len(data_f), num)] #列表划分
		labels = labelGet2(s) #获取标签

		# 2. 读取特征表并归一化
		_, featTable = xlsRead(f"{__file__}/../data/data_{e}.xls", "SharpWave") #读取属性列表
		norm = MinMaxScaler(feature_range=(0,1))  #归一化
		featTable_norm = norm.fit_transform(featTable)

		# 3. 调用scikit-learn模块聚类
		if(not USE_MY_ALGORITHM):
			KM = KMeans(n_clusters=N,random_state=0).fit(featTable_norm)
			labelPre = KM.labels_
			print("Predicted LabelList:\n",labelPre)
			print("Source LabelList:\n",labels)
			print("AdjustedRandIndex:",metrics.adjusted_rand_score(labelPre,labels)) #调节兰德指数

		# 4. 调用手动实现的算法聚类
		else:
			labelOUT, centroids, _ = kmeans(featTable_norm, N)
			print("Predicted LabelList:\n",labelOUT)
			print("Source LabelList:\n",labels)
			print("AdjustedRandIndex:",metrics.adjusted_rand_score(labelOUT,labels)) #调节兰德指数


if __name__ == "__main__":
	main()
	input("按下任意键继续...")