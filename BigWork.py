# -*- coding: utf-8 -*-     支持文件中出现中文字符
#########################################################
'''
Created on 2022-11-15 16:04:57
@author: Zoey
@version: Python 3.10.0
@function: 读取特征, 机器学习分类
@abbreviation:
	+ : 
'''

#########################################################
import LSM
import numpy as np
import matplotlib.pylab as plt
import random
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def Correlation_dis(Ftab, index, N):
	'''	@func: 展示某个特征与标签的相关度
		@para	Ftab:特征矩阵,含标签 
				index:特征索引
				N:抽取的样本数量
		@return: None
	'''
	sleep = {0:[],1:[],2:[],3:[],4:[],5:[]}
	for i in range(len(Ftab)):
		sleep[Ftab[:,-1][i]].append(Ftab[i][index])
	y1 = random.sample(sleep[0],N)
	y2 = random.sample(sleep[1],N)
	y3 = random.sample(sleep[2],N)
	y4 = random.sample(sleep[3],N)
	y5 = random.sample(sleep[4],N)
	y6 = random.sample(sleep[5],N)
	col = ["FuzzyEn","ApEn","SampEn","PermEn"]
	y1.sort(); y2.sort(); y3.sort(); y4.sort(); y5.sort(); y6.sort()
	plt.plot(y1)
	plt.plot(y2)
	plt.plot(y3)
	plt.plot(y4)
	plt.plot(y5)
	plt.plot(y6)
	plt.legend(["Wake","S1","S2","S3","S4","REM"])
	plt.ylabel(col[index])
	plt.show()


def Accuracy(src_label, pre_label):
	'''	@func: 计算各个分期的准确率
		@para	src_label:原始标签 pre_label:预测的标签
		@return: A Matrix
	'''
	OUT = np.zeros((6,4),dtype=int)
	OUT[:,0] = np.array([0, 1, 2, 3, 4, 5]) #第一列为标签取值
	for i in range(len(src_label)):
		OUT[int(src_label[i])][1] += 1 #第二列为原始样本总数
		if(src_label[i]==pre_label[i]):
			OUT[int(src_label[i])][2] += 1 #第三列为正确样本个数
		else:
			OUT[int(src_label[i])][3] += 1 #第四列为错误样本个数
	ACC = OUT[:,2] / OUT[:,1] * 100
	col = ["label", "num_all", "num_T", "num_F", "ACC"]
	print("\t\t".join(col)) 
	print(np.hstack((OUT, ACC.reshape(-1,1))))


def dataExtend(data, label):
	'''	@func: 数据增强
		@para	data:原始数据
				label:需要增强的数据对应的标签
		@return: new data
	'''
	indics = np.argwhere(data[:,-1] == label).ravel()
	small = data[indics]
	# print(small)
	ext_data = np.vstack((data, small))
	return ext_data


# 分类器选择
# CLASSIFIER = "KNN"
CLASSIFIER = "RandomForest"


def main():
	data = LSM.xlsRead(f"{__file__}/../data/feature.xls", "feature")
	data = data[1]  # type: ignore
	dataExt = data
	dataExt = dataExtend(dataExt, 1.0) #数据增强
	dataExt = dataExtend(dataExt, 3.0) #数据增强
	dataExt = dataExtend(dataExt, 5.0) #数据增强
	# Correlation_dis(dataExt,3,20)
	train,test = train_test_split(dataExt, train_size=0.7,random_state=10) #random_state设置为0或者不加，会导致每次训练集划分不同

	if(CLASSIFIER == "KNN"):
		K = 30
		from sklearn.preprocessing import MinMaxScaler #归一化数据
		norm = MinMaxScaler(feature_range=(0,1))  #归一化
		train_norm = norm.fit_transform(train[:,:-1])  # type: ignore
		test_norm = norm.fit_transform(test[:,:-1]) # type: ignore
		nbrs = neighbors.NearestNeighbors(n_neighbors=K,algorithm='kd_tree').fit(train_norm)  #不含标签列
		_, indices = nbrs.kneighbors(test_norm) #拿测试集去试
		from KNN import MostFreq #之前的作业
		labelOUT = [MostFreq(item) for item in train[:,-1][indices]]  # type: ignore
		print("Accuracy of All: ",accuracy_score(test[:,-1], labelOUT),'\n')  # type: ignore
		Accuracy(test[:,-1], labelOUT) # type: ignore

	if(CLASSIFIER == "RandomForest"):
		clf = RandomForestClassifier(n_estimators=200)
		clf.fit(train[:,:-1],train[:,-1])  # type: ignore
		labelOUT = clf.predict(test[:,:-1]) # type: ignore
		print("Accuracy of All: ",accuracy_score(test[:,-1], labelOUT),'\n')  # type: ignore
		Accuracy(test[:,-1], labelOUT) # type: ignore


if __name__ ==  "__main__":
	main()
	input("按任意键继续...")