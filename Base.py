# -*- coding: utf-8 -*-     支持文件中出现中文字符
#########################################################################
"""
Created on Fri Jan 06 10:08:42 2017

@author: Yuyangyou
@revised: Zoey

代码功能描述: (1) 读取Sharp_waves文件,
			(2) 采用巴特沃斯滤波器, 进行1-30Hz滤波
			(3) 画图
			(4) 计算六个特征值
			(5) ...
"""
#####################################################################
from collections import Counter #列表计数
import numpy as np
from scipy import signal # for butt_filter
import math # for log
import matplotlib.pylab as plt #绘图
import xlwt #for excel write
import EntropyHub as EH
import tqdm
from pathos.multiprocessing import ThreadPool as Pool #多线程
###########################################################################################################
# 计算信息熵
def InfoEn(s):
	'''s:需要计算熵的向量
	'''
	list_len = len(s) #总数据长度
	counter = Counter(s) #计算数据列表中数据出现的次数, 最后得到一个字典
	prob = {i[0]:i[1]/list_len for i in counter.items()} # 遍历上述得到的字典, 将key保留, value除以总长得到“概率”, 再组成一个字典
	H = - sum(i[1] * math.log2(i[1]) for i in prob.items()) # type: ignore # 遍历上述字典, 根据-sum(p*log2(p))公式计算熵值
	return H

# 计算近似熵（Approximate Entropy）
def ApEn(s, r = 0.2, m = 2):
	'''s:需要计算熵的向量; r:阈值容限(标准差的系数); m:向量维数
	'''
	s = np.array(s)
	th = r * np.std(s) #容限阈值
	def phi (m):
		N = len(s)
		x = s[ np.arange(N-m+1).reshape(-1,1) + np.arange(m) ]
		# c = np.array([ (( np.abs(x-xi).max(1) <=r).sum()) /(n-m+1) for xi in x ])
		# 似乎没法用 np.frompyfunc 或 np.vectorize
		ci = lambda xi: (( np.abs(x-xi).max(1) <=th).sum()) / (N-m+1) # 构建一个匿名函数
		c = Pool().map (ci, x) #所传递的参数：函数名,函数参数 —— 需要注意, 这里的map函数就自带通函数的特性, 即支持广播机制
		return np.sum(np.log(c)) / (N-m+1)

	return phi(m) - phi(m+1)

# 调用第三方库计算近似熵
def ApEn2(s, r = 0.2, m = 2):
	'''s:需要计算熵的向量; r:阈值容限(标准差的系数); m:向量维数
	'''
	th = r * np.std(s)
	return EH.ApEn(s,m,r=th)[0][-1] #取第一个列表的最后一项

# 计算样本熵（Sample Entropy）
def SampEn(s, r = 0.2, m = 2):
	'''s:需要计算熵的向量; r:阈值容限(标准差的系数); m:向量维数
	'''
	N = len(s)  #总长度
	th = r * np.std(s) #容限阈值

	def Phi(k):
		list_split = [s[i:i+k] for i in range(0,N-k+(k-m))] #将其拆分成多个子列表
		#这里需要注意, 2维和3维分解向量时的方式是不一样的！！！
		Bm = 0.0
		for i in range(0, len(list_split)): #遍历每个子向量
			Bm += ((np.abs(list_split[i] - list_split).max(1) <= th).sum()-1) / (len(list_split)-1) #注意分子和分母都要减1
		return Bm
	## 多线程
	# x = Pool().map(Phi, [m,m+1])
	# H = - math.log(x[1] / x[0]) 
	H = - math.log(Phi(m+1) / Phi(m))
	return H

# 调用第三方库计算样本熵
def SampEn2(s, r = 0.2, m = 2):
	'''s:需要计算熵的向量; r:阈值容限(标准差的系数); m:向量维数
	'''
	th = r * np.std(s) #容限阈值
	return EH.SampEn(s,m,r=th)[0][-1]

# 计算模糊熵(Fuzzy Entropy)
def FuzzyEn(s, r = 0.2, m = 2, n = 2):
	'''s:需要计算熵的向量; r:阈值容限(标准差的系数); m:向量维数; n:模糊函数的指数
	'''
	N = len(s)  #总长度
	th = r * np.std(s) #容限阈值

	def Phi(k):
		list_split = [s[i:i+k] for i in range(0,N-k+(k-m))] #将其拆分成多个子列表
		#这里需要注意, 2维和3维分解向量时的方式是不一样的！！！
		B = np.zeros(len(list_split))
		for i in range(0, len(list_split)): #遍历每个子向量
			di = np.abs(list_split[i] - np.mean(list_split[i]) - list_split + np.mean(list_split,1).reshape(-1,1)).max(1)
			Di = np.exp(- np.power(di,n) / th)
			B[i] = (np.sum(Di) - 1) / (len(list_split)-1) #这里减1是因为要除去其本身, 即exp(0)
		return np.sum(B) / len(list_split)
	# 多线程
	# x = Pool().map(Phi, [m,m+1])
	# H = - math.log(x[1] / x[0]) 
	H = - math.log(Phi(m+1) / Phi(m))
	return H

# 调用第三方库
def FuzzyEn2(s:np.ndarray, r = 0.2, m = 2, n = 2):
	'''s:需要计算熵的向量; r:阈值容限(标准差的系数); m:向量维数; n:模糊函数的指数
	'''
	th = r * np.std(s)
	return EH.FuzzEn(s, m, r=(th, n))[0][-1]

# 将数据写入表格
def xlsWrite(filename:str, sheetname:str, A, col:list[str]|tuple[str,...]): 
	'''filename:写入的文件名; A:需要写入表格的数据; col:列名; sheetname:表格名
	'''
	m = len(A) #矩阵长度
	n = len(A[0]) #矩阵列数
	workbook = xlwt.Workbook(encoding='utf-8', style_compression=0)
	sheet = workbook.add_sheet(sheetname, cell_overwrite_ok=True)
	for i in range(m+1):
		for j in range(n):
			if(i == 0): sheet.write(i,j,col[j])
			else: sheet.write(i,j,str(A[i-1][j]))
	workbook.save(filename)

# 将数据以表格形式写入到txt文件
def txtWrite(filename:str, A, col:str):
	'''filename:需要写入的文件名; A:需要写入的数据; col:列名
	'''
	with open(filename, 'w') as file:
		file.write(col + "\n")
		for i in range(len(A)):
			file.write(str(A[i]).replace("[",'').replace(']','').replace(' ','\t') + '\n')

# 打开给定数据文件, 并读取一列数据
def getdata(filename, k): 
	'''filename:读取的文件名; k:所取数据的列数,0~3
	'''
	with open(filename) as file:    #使用with打开文件, 不过是否读取成功, 都会自动关闭, 更加安全
		A = file.read().split( )    #划分, 似乎是按空白字符划分, 而不是空格
		A = [float(i) for i in A]   #转换为浮点数
		s = A[0:45000*4+k:4]        #取其中一列, 其结构为: [初始值:终止值:步长]
	return s

# 巴特沃斯带通滤波器
def butter_band_filter(s, fs = 3000, cutoff:list = [1,30], order = 2):
	'''s:需要滤波的数据; fs:采样频率; cutoff:截止频率; order:阶数
	'''
	lowcut = cutoff[0]
	highcut = cutoff[1]
	nyq = 0.5*fs                                        #设立采样频率变量nyq, 采样频率=fs/2。
	low = lowcut/nyq
	high = highcut/nyq
	b,a = signal.butter(order,[low,high],btype='band')  #设计巴特沃斯带通滤波器 “band”
	s_filter = signal.lfilter(b,a,s)                    #将s带入滤波器, 滤波结果保存在s_filter中
	return s_filter

# 特征提取
def getFeature(dataTable:np.ndarray|list) -> tuple[np.ndarray,list[str]]: 
	''' @func: 提取给定的特征, 调整特征在这个函数中调整
		@para dataTable: 数据列表, 每一行代表一个样本
		@return: 一个特征表及特征名称
	'''
	dataTable = np.array(dataTable) #作用是去掉外层的括号
	num = len(dataTable)#子列表的个数
	# funcdict: 特征函数字典, key为特征名称, value为特征函数, 添加直接增加一行即可, 删除直接注释某一行即可
	funcdict = 	{
				# "Mean":lambda s: np.mean(s), #均值
				# "Var":lambda s: np.var(s), #方差
				# "Std":lambda s: np.std(s), #标准差
				# "Range":lambda s: s.max() - s.min(), #极差 
				"ApEn":lambda s: ApEn(s,0.2,2), #近似熵
				"SampEn":lambda s: SampEn(s,0.2,2), #样本熵
				"FuzzyEn":lambda s: FuzzyEn2(s, 0.2, 2, 2), #模糊熵
				# "Max":lambda s: np.max(s) #最大值
				}
	F = np.zeros((num,len(funcdict))) #特征矩阵
	featName = list(funcdict.keys()) #特征名称
	for i in tqdm.trange(num):
		j = 0
		for item in funcdict:
			F[i][j] = funcdict[item](dataTable[i])
			j += 1
	return F,featName


# 主函数
def main():
	print("Please Wait for a few seconds...")
	np.set_printoptions(linewidth=140) #设置打印一行的长度
	for i in tqdm.trange(113, 115):
		s = getdata(f"{__file__}/../data/20151026_{i}", 0) #读取数据第1列
		# fig1 = plt.figure("SrcData")
		# plt.plot(s)
		sf = butter_band_filter(s)   #滤波
		# fig2 = plt.figure("FilterData")
		# plt.plot(s1)
		# plt.show()
		singleNum = 150 #单个列表中包含数据点的个数
		Table = [sf[i:i+singleNum] for i in range(0, len(sf), singleNum)] #列表划分
		F,col = getFeature(Table) # 特征提取
		print('\nThe Feature Table of',i,'is:\n',col, '\n', F)
		xlsWrite(f"{__file__}/../data/data_{i}.xls", "SharpWave", F, col)  #将数据写入excel


if __name__ == "__main__":
	main()
	input("按下任意键继续...")