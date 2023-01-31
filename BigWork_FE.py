# -*- coding: utf-8 -*-     支持文件中出现中文字符
#########################################################
'''
Created on 2022-11-15 16:05:57
@author: Zoey
@version: Python 3.10.0
@function: data preprocess and feature engineering
@abbreviation:
	+ : 
'''
#########################################################
import Base
import numpy as np
import tqdm
import pywt
import EntropyHub as EH
import matplotlib.pylab as plt

def show(a,b):
	'''	@func: 展示滤波前后的数据
		@para	a&b: 滤波前后的数据(行向量)
		@return: None
	'''
	plt.figure(0,(12,6))
	plt.subplot(2,1,1) #上下两个图,第一个
	plt.plot(a)
	plt.title("Raw Data")
	plt.subplot(2,1,2)
	plt.plot(b)
	plt.title("Denoised Data")
	plt.subplots_adjust(hspace=0.5)
	plt.show()


def main():
	dataset = np.loadtxt(f"{__file__}/../data/sc4002e0_data.txt") #读取原始数据
	dataset = dataset.reshape(-1,3000) #帧划分
	label = np.loadtxt(f"{__file__}/../data/sc4002e0_label.txt") #标签
	label = label[:-1] #去除末尾数据
	for i in range(len(label)): #去除运动数据
		if(label[i] == 6.0): 
			# print(i)
			label = np.delete(label, i)
			dataset = np.delete(dataset, i, axis=0)
			break
	# print(dataset)

	FTab = np.zeros((len(dataset),4)) #特征表
	for i in tqdm.trange(len(dataset)): #数据处理与特征提取
		# 1 小波分解
		coeffs = pywt.wavedec(dataset[i],'db4') #小波分解(decompose)得到小波系数
		threshold = 0.43 #设定阈值
		for j in range(4,len(coeffs)):
			coeffs[j] = pywt.threshold(coeffs[j], threshold*max(coeffs[j]), mode='soft')  # 软阈值分隔函数
		datarec = pywt.waverec(coeffs,'db4') #小波重建(recompose)，得到滤波后的数据
		# show(dataset[i],datarec)
		dataset[i] = datarec

		# 2 特征提取
		FTab[i,0] = EH.FuzzEn(dataset[i],m=2,r=(0.2*np.std(dataset[i]),2))[0][-1]
		FTab[i,1] = EH.ApEn(dataset[i],r=0.2*np.std(dataset[i]))[0][-1]
		FTab[i,2] = EH.SampEn(dataset[i],r=0.2*np.std(dataset[i]))[0][-1]
		FTab[i,3] = EH.PermEn(dataset[i],m=5,tau=1,Logx=2,Norm=True)[0][-1]

	ALL = np.hstack((FTab, label.reshape(-1,1)))
	col = ("FuzzyEn", "ApEn", "SampEn", "PermEn","Label")
	Base.xlsWrite(f"{__file__}/../data/feature.xls","feature",ALL, col)

if __name__ ==  "__main__":
	main()
	input("请按任意键继续...")