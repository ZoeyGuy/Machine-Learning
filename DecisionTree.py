# -*- coding: utf-8 -*-     支持文件中出现中文字符
#########################################################################
"""
Created on 2022.10.11
@author: Zoey
@version: Python 3.10
@简写说明
  + feat: feature
  + calc: calculate
  + Vec: Vector
  + chd: child
  + 
  + 
  + 

"""
#########################################################################
import Base #第一次作业, 基本数据处理及特征提取
from LSM import xlsRead #第二次作业, 读取数据
import numpy as np
from sklearn import tree
from scipy import signal
import math
import matplotlib.pylab as plt
#########################################################################

def labelGet1(s:list|np.ndarray) -> np.ndarray:
	''' @func: 根据每一段曲线的形状来赋予标签
		@para s:划分好的列表, 每一行代表一个子数据
		@return: 一个标签list, 长度等于s的行数
	'''
	n = len(s) #划分子列表数
	label = np.zeros(n,dtype="int") #给各个子列表分配的标签
	for i in range(n):
		s2 = Base.butter_band_filter(s[i], cutoff=[1,20]) #滤波处理
		z = np.polyfit(range(len(s2)),s2,3) #三次多项式拟合
		p = np.poly1d(z)
		s3 = p(range(len(s2)))
		# peaks = signal.find_peaks(s3,distance=10) #另一种求极值的方法
		peak_max = list(signal.argrelextrema(s3,np.greater)[0]) #极大值索引 列表
		peak_min = list(signal.argrelextrema(-s3, np.greater)[0]) #极小值索引 列表
		# print(signal.find_peaks_cwt)
		# print([peak_max,peak_min])
		if(peak_max == [] and peak_min == []): #没有波峰和波谷
			label[i] = 1
		elif(peak_min != [] and peak_max == []):#只有一个波谷
			label[i] = 2
		elif(peak_max != [] and peak_min == []):#只有一个波峰
			label[i] = 2
		elif(peak_max > peak_min):#有一个波峰一个波谷, 且波峰在前
			label[i] = 3
		elif(peak_max < peak_min):#有一个波峰一个波谷, 且波峰在后
			label[i] = 3
		else:#其他类, 一般不会出现
			label[i] = 0
		print(label[i])
		# print(np.polyder(p,1).r)
		# poles_num = np.sum(abs(dif - 0) <= 0.000001)
		# print(poles_num)
		plt.plot(s2,'k')
		plt.plot(s3,'r')
		# print(np.mean(s2), max(s2))
		plt.show()
		plt.close()
	return label


def labelGet2(s:list|np.ndarray)->np.ndarray:
	''' @func: 根据sharp wave曲线的特征来分配标签: 上升段, 下降段, 平稳段
		@para s: 划分好的列表
		@return: 一个标签list, 长度等于s的行数
	'''
	n = len(s) #划分子列表数
	labels = np.zeros(n,dtype="int") #给各个子列表分配的标签
	for i in range(n): #按均值来分配标签
		if(np.mean(s[i]) >= 0.004): labels[i] = 1
		elif(np.mean(s[i]) <= -0.006): labels[i] = -1
	return labels


def DataSetExtract(dataset:np.ndarray, featIndex, featValue) -> np.ndarray:
	''' @func: 在给定数据集(特征+标签)中提取出某个特征为某个特定值的子数据集
		@para  dataset: 数据集, 行为样本数, 列包含特征和标签
				featIndex: 目标特征在数据集中的索引
				featValue: 目标特征的一个取值
		@return: 划分后的子数据集
	'''
	SampleNum = len(dataset) #样本数
	featList = dataset[:, featIndex] #dataset中该列特征的所有取值
	col = [] #符合特征值的样本的索引
	for i in range(SampleNum):
		if(featList[i] == featValue): col.append(i)
	subDataset = np.delete(dataset[col],featIndex,1) #取对应行, 并删除特征列
	return subDataset


def FreqCalc(A:np.ndarray, index) -> dict:
	''' @func: 获取矩阵A的某一列(属性or标签)中各个取值的频次
		@para A: 对象矩阵
			  index: 要取的列的索引
		@return: a dict
	'''
	Value = A[:, index]

	unique_value,occurrence = np.unique(Value, return_counts=True)
	Dict = dict(zip(unique_value,occurrence))

	# uniqueValue = set(Value) #这一列的不同取值
	# uniqueNum = len(uniqueValue) #取值有多少种情况
	# Dict = dict(zip(uniqueValue,np.zeros(uniqueNum, dtype=int))) #构建一个字典, value(标签对应的样本个数)初值全为0
	# for item in Value:
	# 	Dict[item] += 1
	
	return Dict
	


def InfoEntropy(dataset:np.ndarray) -> float:
	''' @func: 计算某个数据集的标签信息熵
		@para dataset: 数据集, 行为样本数, 列为特征+标签
		@return: 数据集的信息熵
	'''
	labelDict = FreqCalc(dataset,-1) #取最后一列
	EN = 0.0
	for item in labelDict: #遍历标签字典——自动排除了样本数为零的标签
		tmp = float(labelDict[item]) / len(dataset)
		EN += - tmp * math.log2(tmp)
	return EN


def GiniRatio(dataset:np.ndarray) -> float:
	'''	@func: 计算基尼指数
		@para  dataset:数据集
		@return: Gini Ratio
	'''
	labelDict = FreqCalc(dataset,-1) #取最后一列
	Gini = 0.0
	for item in labelDict: #遍历标签字典——自动排除了样本数为零的标签
		tmp = float(labelDict[item]) / len(dataset)
		Gini += tmp**2
	return 1 - Gini


def EntropyGain(dataset:np.ndarray, featIndex) -> float:
	''' @func: 计算按某个特征分类得到的熵增益
		@para  dataset: 数据集, 行为样本数, 列为特征+标签
				featIndex: 目标特征在数据集中的索引
		@return: 熵增益, 浮点类型
	'''
	EN0 = InfoEntropy(dataset) #分类前的数据集
	featDict = FreqCalc(dataset, featIndex) #特征取值字典
	EN1 = 0.0
	for item in featDict:
		subDataset = DataSetExtract(dataset,featIndex,item)
		EN1 += InfoEntropy(subDataset) * (featDict[item] / len(dataset)) #各个子数据集所占比例乘以各自的熵增益
	return EN0 - EN1


def EntropyGainRate(dataset:np.ndarray, featIndex) -> float:
	'''	@func: 计算特征的熵增益率
		@para  dataset: 数据集
			   featIndex: 特征索引
		@return: 熵增益率
	'''
	ENG = EntropyGain(dataset,featIndex) #熵增益
	dic = FreqCalc(dataset,featIndex) #该特征对应的各个取值形成的字典
	num_sum = sum(list(dic.values()))
	Q = 0.0
	for item in dic:
		tmp = dic[item] / num_sum
		Q += - tmp * math.log2(tmp)
	# print(ENG/Q if Q else ENG)
	return ENG/Q if Q else ENG #如果tmp为1(特征只有一个取值)，则Q为0，要返回原熵增益



def GetBestFeat(dataset:np.ndarray):
	''' @func: 遍历当前数据集下各个特征, 然后取熵增益最大的特征
		@para  dataset:当前分类下的数据集
		@return: the index of the beat feature 
	'''
	featNum = len(dataset[0]) #当前数据集下剩余特征的个数+1
	MAX_EN = 0.0; INDEX = 0 #最大熵增益的特征及其对应的索引
	for i in range(featNum-1):
		EN_tmp = EntropyGain(dataset,i) #利用熵增益来选择特征
		# EN_tmp = EntropyGainRate(dataset,i) #利用熵增益率来选择特征
		if(EN_tmp > MAX_EN): MAX_EN = EN_tmp ; INDEX = i
	# print(MAX_EN)
	if(int(MAX_EN * 100) == 0): return -1 #剩下的特征熵增益基本接近零了, 分类就可以结束了, 通过调整100来调整精度
	else: return INDEX


def MajorFeature(labelList:np.ndarray):
	''' @func: 当特征列都被删完了只剩下标签列, 但是仍然有不同的标签, 
			   此时只能取样本数较多的标签作为叶子节点
		@para labelList: 最后的标签列表, 要求是列向量, 否则会报错
		@return: A label as leaf node
	'''
	labelDict = FreqCalc(labelList,0)
	sortedDict = sorted(labelDict.items(), key=lambda d:d[1]) #按value排序,升序
	# import operator #利用operator库中的函数
	# sortedDict = sorted(labelDict,key=operator.itemgetter(1),reverse=True)
	return sortedDict[-1][0] #返回最后一个, 样本数最多的一个标签


def FeatDiscrete(srcDataSet:np.ndarray, colIndex = ..., classNum:int = 2) -> np.ndarray:
	''' @func: 采用C4.5中处理连续值的方式, 对连续值进行离散二分类
		@para srcDataSet: 原始数据集, 特征矩阵
			  colIndex: 需要处理的特征列的索引, 是一个索引列表, 默认取所有特征列（除标签列）
			  classNum: 分类数目, 默认二分类
		@return: dstDataSet 
	'''
	def MidNumList(srcList:np.ndarray) -> np.ndarray:
		''' @func: 提取一个列表的相邻元素中位数列表, n个数可以得到n-1个中位数
			@para srcList: 原始数据列表
			@return: dstList
		'''
		sortedList = np.sort(srcList) #升序排列
		dstList = np.array([])
		for i in range(len(sortedList)-1):
			dstList = np.append(dstList, np.median(sortedList[i:i+2]))
		return dstList

	def GetCombine(MidList:list|np.ndarray, Num = 2) -> list:
		''' @func: 根据中位数列表取几个分界点
			@para MidList: 中位数列表
				  Num: 取点的个数
			@return: 取的关键点形成的列表, 是一个二维列表
		'''
		import itertools as it #求排列组合
		CombineList = [e for e in it.combinations(MidList, Num)]
		return CombineList

	def discrete(srcDataVec:list|np.ndarray, CombineList:list):
		''' @func: 根据分界点列表, 分配离散值, 从0开始
			@para srcDataVec: 原始特征向量, 全部是连续值 
				  CombineList: 分界点列表中的一行, 即一种情况
			@return: 特征的离散值, 长度与原数据长度相同
		'''
		length = len(srcDataVec)
		discreteOut = - np.ones(length) #初始值全为-1
		for i in range(length):
			for j in range(len(CombineList)):
				if(srcDataVec[i] <= CombineList[j]): discreteOut[i] = j
			if(discreteOut[i] == -1): discreteOut[i] = len(CombineList) #大于所有值的情况
		return discreteOut

	if(colIndex == ...): colIndex = range(srcDataSet[0].size - 1) #特征数, 即列数
	for i in colIndex: #遍历选择的每一列
		featList = srcDataSet[:,i].copy() #取某一列数值 记得要复制一份
		midlist = MidNumList(featList) #中位数列表
		CombineList = GetCombine(midlist,classNum-1) #分界点列表
		# print(CombineList)
		# print(len(CombineList))
		EN_GAIN_MAX = 0.0 ; DEVIDE_LIST = [] #最大熵增益及对应的分界点
		for item in CombineList:
			srcDataSet[:,i] = discrete(featList,item) #注意copy的问题
			tmp = EntropyGain(srcDataSet,i) #某一种情况的熵增益
			if(tmp>EN_GAIN_MAX): EN_GAIN_MAX = tmp; DEVIDE_LIST = item			
		srcDataSet[:,i] = discrete(featList, DEVIDE_LIST) #确定分离点
	return srcDataSet


def CreateTree(dataset:np.ndarray, featList) -> dict:
	''' @func: 根据输入的特征矩阵和标签向量构造决策树【核心函数】
		@para  dataset: 传入的样本数据, 即特征+标签
				featList: 特征列表, 存储特征的名称
		@return: A tree dict
	'''
	labelList = dataset[:,-1].copy() #取最后一列标签
	if(len(dataset[0]) == 1): return MajorFeature(dataset) #特征都删除了, 只剩下标签, 取值最多的种类
	if(np.unique(labelList).size == 1): return labelList[0] #分到叶子节点, 返回标签值
	BestFeatIndex = GetBestFeat(dataset)
	if(BestFeatIndex == -1): return MajorFeature(dataset) #特征增益都为0了, 没必要再分下去了
	BestFeat = featList[BestFeatIndex]
	Tree = {BestFeat:{}} #用字典来构造一个树, 其基本单元的结构为{feat:{featValue1:{chdTree1},featValue2:{chdTree2}}}
	BestFeatValue = np.unique(dataset[:,BestFeatIndex])
	for item in BestFeatValue: #遍历特征的每个取值
		subDataSet = DataSetExtract(dataset,BestFeatIndex,item)
		Tree[BestFeat][item] = CreateTree(subDataSet, np.delete(featList,BestFeatIndex,0))
	return Tree


def Predict(tree:dict, testDataset:np.ndarray, featList:list):
	''' @func: 根据得到的树来预测某个样本的标签取值
		@para tree: 已经创建好的一个树, 是一个字典
			  testDataset: 测试集
			  featList: 特征列表
		@return: 预测出的标签取值
	'''
	if(testDataset.ndim == 1): testDataset = testDataset.reshape(1,-1) #只有一行数据, 就手动套一层括号
	sampleNum, _ = testDataset.shape #确定样本个数（行）
	labelPre = np.zeros(sampleNum,testDataset.dtype) #预测的标签值, 一个列表，考虑到
	for i in range(sampleNum): #遍历每一条数据
		testVec = testDataset[i] #取测试集中的这一行
		firstFeat = list(tree.keys())[0] #需要先转换为一个list
		featIndex = featList.index(firstFeat) #特征在特征表中的索引
		actualValue = testVec[featIndex] #输入的样本数据中该位置特征的实际取值
		subTree = tree[firstFeat][actualValue] #第二级的子树
		if isinstance(subTree, dict): #如果子树还是一个字典, 则需要继续剥
			labelPre[i] = Predict(subTree, testVec, featList)[0] #注意返回的也是一个列表，所以需要去掉括号
		else:  #子树剥到头, 变成叶子节点, 则其值即为标签取值
			labelPre[i] = subTree
	return labelPre


def ACC_Calc(src:np.ndarray|list, pre:np.ndarray|list) -> float:
	'''	@func: 计算分类的准确率
		@para src:原标签列表
			  pre:预测的标签
		@return: 准确率
	'''
	src = np.array(src); pre = np.array(pre)
	return np.sum(src == pre) / src.size


def TreePlot(tree:dict, layer = 0) -> None:
	'''	@func: 以更易读的方式展示字典形式的树
		@para tree: 树字典
		@return: None
	'''
	firstFeat = list(tree.keys())[0] #需要先转换为一个list
	secondTree = tree[firstFeat] #二级树
	valuelist = list(secondTree.keys()) #第一级特征的取值
	for item in valuelist:
		chdtree = secondTree[item]
		if isinstance(chdtree, dict):
			print('|\t'*layer, firstFeat, item, ' : ')
			TreePlot(chdtree, layer+1)
		else: print('|\t'*layer, firstFeat, item, ' : ', chdtree)


# USE_MY_TREE = 1 # 决定是否使用自己写的决策树代码
USE_MY_TREE = 0


# 主函数
def main():
	print("Please Wait for a few seconds...\n")
	for e in range(113,115):
		# 1. 数据预处理, 得到标签
		data = Base.getdata(f"{__file__}/../data/20151026_{e}", 0) #读取第一列的数据
		data_f = Base.butter_band_filter(data) #滤波
		num = 150 #划分的子列表数
		s = [data_f[i:i+num] for i in range(0, len(data_f), num)] #列表划分
		labels = labelGet2(s) #获取标签

		# 2. 读取特征表, 并加上标签构成数据集
		cols, featTable = xlsRead(f"{__file__}/../data/data_{e}.xls", "SharpWave") #读取属性列表
		dataset = np.hstack((featTable,np.array([labels]).T)) #注意维度关系
		dataset = FeatDiscrete(dataset,classNum=2) #将连续值离散化, 确定分几类
		# print(dataset)
		# Base.xlsWrite(f"{__file__}/../out.xls",'OUT2',dataset,cols+["label"])

		# 3. 训练测试集划分
		train = 0.8; test = 1-train #训练与测试集的分配
		# 方式一
		# index = np.zeros(len(dataset),dtype=int) #或者直接生成bool数组
		# index[0:int(train*len(dataset))] = 1
		# np.random.shuffle(index)
		# trainSet = dataset[index.astype(bool)] #注意这里要用bool类型的，否则花式索引会出错
		# testSet = dataset[(1-index).astype(bool)]
		# 方式二
		# index = np.arange(len(dataset),dtype=int)
		# np.random.shuffle(index)
		# trainSet = dataset[index[:int(train*len(dataset))]]
		# testSet = dataset[index[int(train*len(dataset)):]]
		# 方式三
		trainSet = dataset[:int(train*len(dataset))] #训练集
		testSet = dataset[int(train*len(dataset)):] #测试集

		# 4. 调用自己写的树进行训练并预测
		if(USE_MY_TREE):
			treedic = CreateTree(trainSet,cols) #构建树，生成树字典
			print('Tree Structure of',e,':'); TreePlot(treedic) #画出树字典
			labelOUT = Predict(treedic,testSet,cols) #根据树字典测试测试集
			print("Predicted LabelList:\n",labelOUT)
			print("Source LabelList:\n",testSet[:,-1])
			print("Accuracy: ",ACC_Calc(testSet[:,-1], labelOUT),'\n')
		
		# 5. 调用第三方库进行训练并预测
		else: 
			clf = tree.DecisionTreeClassifier()
			clf.fit(trainSet[:,:-1],trainSet[:,-1]) #拆分特征和标签
			tree.plot_tree(clf)
			labelOUT = clf.predict(testSet[:,:-1])
			print('Predicted LabelList:\n',labelOUT)
			print('Source LabelList:\n',testSet[:,-1])
			print("Accuracy: ",ACC_Calc(testSet[:,-1], labelOUT),'\n')


if __name__ == "__main__":
	main()
	input("按下任意键继续...")
