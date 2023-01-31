# -*- coding: utf-8 -*-     支持文件中出现中文字符
#########################################################################
"""
Created on 2022.10.11
@author: Zoey
python version:3.10.8

注意: 运行前要先将生成的xls文件放在该文件同目录下
"""
#####################################################################
import numpy as np
import xlrd
import matplotlib.pylab as plt

# 按行读取excel表中的特征值
def xlsRead(filename, sheetname)->tuple[list[str],np.ndarray]: 
	'''filename:文件名; sheetname:表格名
	'''
	workbook:xlrd.Book = xlrd.open_workbook(filename)
	sheet = workbook.sheet_by_name(sheetname)
	# print(sheet.name)
	rows = sheet.nrows #获取表格有效行数
	cols = sheet.ncols #获取表格有效列数，即特征数
	A = np.zeros((rows-1,cols)) #去除第一行题头
	col = sheet.row_values(0) #得到特征题头
	for i in range(rows-1):
		A[i] = sheet.row_values(i+1)
	return col, A

# 最小二乘法进行线性回归
def LSM(X:np.ndarray, Y:np.ndarray): 
	'''X: m行n+1列的矩阵(m为样本个数,n为特征个数),本实例中为时间轴,“单特征”
	   Y: m行1列的矩阵
	   return: w为n+1行1列的矩阵(包含w0)
	   本实例中,m=15,n=1(即时间序列,为单特征)
	'''
	w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T),Y)
	return w


DISP_PIC = 0
# DISP_PIC = 1


# 主函数
def main():
	for e in range(113, 115):
		title, fea = xlsRead(f"{__file__}/../data/data_{e}.xls", "SharpWave") #读取6列特征值
		m = len(fea) #样本数m
		n = len(title) #特征数
		R2 = np.zeros(n) #R方评价指标
		for i in range(n): #遍历所有特征
			y = fea[:,i] #取特征的一列
			x = np.vstack((np.ones(m),np.arange(1,m+1))).T # 15行2列
			b,a = LSM(x,y) #y=ax+b
			y2 = a*x[:,1]+b
			Var = np.var(y)
			MSE = np.mean((y-y2)**2)
			R2[i] = 1 - MSE/Var #计算R方评价指标
			print("The R2Score of",title[i],'in',e,'is:',R2[i])
			y3 = a*(m+1)+b #预测值
			# print(LSM(x,y))
			fig = plt.figure()
			plt.plot(x[:,1],y)
			plt.plot(x[:,1],y2,'r')
			plt.plot([m,m+1],[y[-1],y3],color='k')
			plt.plot(m+1,y3,marker='.',color='k')
			plt.title(title[i])
			import os
			if(not os.path.exists("./img")):
				path = os.mkdir("./img")
			plt.savefig(f"{__file__}/../img/{e}" + title[i]) #保存图片
			if(DISP_PIC): plt.show()
			# plt.close('all')
	

if __name__ == "__main__":
	main()
	input("按下任意键继续...")