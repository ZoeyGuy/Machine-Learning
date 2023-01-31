import numpy as np
from numpy import ndarray as ndary
import sys, os
from tqdm import tqdm
import warnings; warnings.filterwarnings ("ignore", module="tqdm")


# region class DecisionTree
class DecisionTree:
	# tbl(data), row(item), col(attr)+lastcol(categ), val+lastcolval, uniqueVal(case)+uniquelastcolval(tag)

	class Node:

		def __init__ (self, itms:ndary=np.array([[]]), attr:ndary|int=-1, categ:str="",
				chdn:dict[str,"DT.Node"]={}) ->None:
			# 包含的项目（行号列表），划分前剩余属性（列号列表）或用来划分的属性（列号），最终类别（取值），划分形成的子节点（取值到节点的映射）
			(self.itms, self.attr, self.categ, self.chdn) = (itms, attr, categ, chdn)

	def __init__ (self, attr:ndary, data:ndary) ->None:
		# 属性（列的表头），数据（二维表格），决策树根（节点）。约定属性和数据的最后一列是最终分类标签
		(self.attr, self.data, self.root) = ( attr, data, DT.Node (
			np.arange(data.shape[0]), np.arange(data.shape[1]-1) ))

	@staticmethod
	def countFreq (vals:ndary) ->tuple[ndary,ndary]:
		''' 统计列表vals内值的种类及其出现的频率。保持出现顺序
		'''
		keysSrt, idxs, cntsSrt = np.unique (vals, return_index=True, return_counts=True)
		keys = keysSrt [(odr:= np.argsort (idxs))]
		freqs = (cntsSrt / np.sum (cntsSrt)) [odr]
		return (keys, freqs)

	@staticmethod
	def entropy (probs:ndary) ->float:
		# 输入概率列表，输出信息熵
		return - np.sum (probs * np.log2 (probs))  # type: ignore

	@staticmethod
	def group (tbl:ndary, col:int, rtnIdx:bool=False) ->tuple[ndary,list[ndary]]:
		''' 对第col列的每种取值，将二维表tbl的行分组，返回取值种类和分组结果，结果可选是取值还是行号。保持出现顺序
		'''
		keysSrt, idxs = np.unique (tbl[:,col], return_index=True)
		# keys = keysSrt [np.argsort (idxs)]
		keys = tbl [sorted(idxs), col]
		# return list(np.frompyfunc( lambda k: tbl[tbl[:,col]==k] ,1,1)( keys ))
		return (keys, list(map( lambda k: tbl[tbl[:,col]==k] , keys ))) if not rtnIdx \
		else (keys, list(map( lambda k: np.argwhere(tbl[:,col]==k).flatten() , keys )))

	@staticmethod
	def gain (vals:ndary, categs:ndary) ->float:
		# 输入数据表在 用于划分的属性和最终分类属性 上的值列表，输出信息增益
		newEnts = [ DT.entropy(DT.countFreq(g[:,-1])[1])
			for g in DT.group(np.vstack((vals,categs)).T,0)[1] ]
		return DT.entropy(DT.countFreq(categs)[1]) - np.sum(DT.countFreq(vals)[1]*newEnts)  # type: ignore

	@staticmethod
	def most (vals:ndary):
		''' 找到列表vals内出现次数最多的元素中最先出现的
		'''
		keysSrt, idxs, cnts = np.unique (vals, return_index=True, return_counts=True)
		return keysSrt [(odr:= np.argsort (idxs))] [np.argmax (cnts [odr])]

	def train (self, maxDepth:int) ->"DT":
		# 训练
		def split (node:DT.Node) ->bool:
			# 尝试将指定节点划分开
			if type(node.attr)==int: return False  # type: ignore 
			if len(node.attr)==0: node.attr = -1; return False  # type: ignore
			data = self.data[node.itms]
			gains = [DT.gain (data[:,a], data[:,-1]) for a in node.attr]  # type: ignore
			if np.max(gains)==0: node.attr = -1; return False
			bestAttr:int = node.attr[ np.argmax(gains) ]  # type: ignore
			node.chdn = {
					key: DT.Node (
						node.itms[grp],
						np.delete (node.attr, np.argmax(gains)),
						DT.most (self.data[node.itms[grp],-1]) )
				for key,grp in zip(*DT.group (data, bestAttr, True)) }
			node.attr = bestAttr; return True
		def _train (node:DT.Node, restDepth:int) ->None:
			if restDepth<=0: node.attr = -1; return
			if split (node):
				pbar.update (1)
				for chdC,chdN in node.chdn.items():
					_train (chdN, restDepth-1)
			else: pbar.update (splitNumEst(restDepth))  # type: ignore

		# 以下几行代码仅用于估算进度条总量
		attrNum = self.data.shape[1]-1
		caseNumAvg = np.mean([ len(DT.countFreq(self.data[:,i])[0]) for i in range(attrNum) ])
		realMaxDepth = min(maxDepth,attrNum)
		splitNumEst = lambda restDepth: float(f"{ (caseNumAvg**(restDepth)-1)/(caseNumAvg-1) :.2f}")
		pbar = tqdm (desc="Training", total=splitNumEst(realMaxDepth), leave=False, unit="split")
		self.root.categ = DT.most (self.data[:,-1])
		_train (self.root, realMaxDepth)  # 直接传maxDepth也能运行，但将无法估计进度
		pbar.close()
		return self

	def toStr (self) ->str:
		# 树转字符串
		categs = DT.countFreq(self.data[:,-1])[0]
		def _toStr (attr:str, case:str, node:DT.Node, lvl:int) ->str:
			return "\n".join (
				( ["|   "*lvl+f"{attr} = {case}: {node.categ} [{'/'.join([str(np.sum(self.data[node.itms,-1]==categ)) for categ in categs])}]"]
				if lvl!=0 else
				[f"ROOT: {node.categ} [{' / '.join([f'{np.sum(self.data[node.itms,-1]==categ)} {categ}' for categ in categs])}]"] )
				+ [_toStr(self.attr[node.attr],chdC,chdN,lvl+1) for chdC,chdN in node.chdn.items()] )
		return _toStr ("", "", self.root, 0)

	def predict (self, itm:ndary) ->str:
		# 预测单个项目的类别
		cur = self.root
		while (case:= itm[cur.attr]) in cur.chdn:
			cur = cur.chdn[case]
		return cur.categ

	@staticmethod
	def errR8 (pred:ndary, actl:ndary) ->float:
		# 输入预测值序列和实际值序列，输出错误率
		return np.sum(pred!=actl) / len(pred)

	def test (self, data:ndary) ->tuple[list[str],float]:
		# 测试
		return ( pred:= [self.predict(itm) for itm in data],
			DT.errR8 (np.array(pred), data[:,-1]) )

DT = DecisionTree
# endregion


sys.argv[1:] = (lambda name, depth, rvs:
		[*[f"./data/{name}_train.tsv", f"./data/{name}_test.tsv"][::-1 if rvs else 1], f"{depth}", *[f"./out/{name}_train.lbl", f"./out/{name}_test.lbl"][::-1 if rvs else 1], f"./out/{name}_metrics.txt"]) \
	(["small", "politicians", "education", "mushroom"] [0], 2, False)

if __name__ == "__main__":
	try: _, trainInPath, testInPath, maxDepth, trainOutPath, testOutPath, metricsOutPath, *_ = sys.argv
	except: print ("Usage: decisionTree.py <trainIn:tsv> <testIn:tsv> <maxDepth:int> <trainOut:lbl> <testOut:lbl> <metricsOut:txt>"); exit(1)

	trainTbl:ndary = np.loadtxt (trainInPath, dtype=str, comments=None, delimiter="\t", converters=None, skiprows=0, usecols=None, unpack=False)  # type: ignore
	trainAttr, trainData = trainTbl[0], trainTbl[1:]
	dTree = DecisionTree (trainAttr, trainData) .train (int(maxDepth))
	print (dTree.toStr())
	trainPred, trainErrR8 = dTree.test (trainData)
	testData:ndary = np.loadtxt (testInPath, dtype=str, comments=None, delimiter="\t", converters=None, skiprows=1, usecols=None, unpack=False)  # type: ignore
	testPred, testErrR8 = dTree.test (testData)
	os.makedirs (os.path.dirname (trainOutPath), exist_ok=True)
	with open (trainOutPath, "w") as file:
		file.write ("\n".join (trainPred) +"\n")
	os.makedirs (os.path.dirname (testOutPath), exist_ok=True)
	with open (testOutPath, "w") as file:
		file.write ("\n".join (testPred) +"\n")
	os.makedirs (os.path.dirname (metricsOutPath), exist_ok=True)
	with open (metricsOutPath, "w") as file:
		file.write (f"error(train): {trainErrR8}\nerror(test): {testErrR8}\n")


# # region test
# 	n = np.array ([
# 		["a","1","F"],
# 		["b","2","F"],
# 		["c","3","F"],
# 		["c","3","T"],
# 		["b","3","T"],
# 		["b","3","T"] ])

# 	print (DecisionTree.most (np.array(list("bcccaaddd"))))
# 	print (DecisionTree.countFreq (n[:,0]))
# 	print (DecisionTree.group (n,0,True))
# 	print (DecisionTree.gain (n[:,0],n[:,-1]))  # 0.20751874963942196

# 	N = DecisionTree( np.array(["char","int","bool"]), n ) .train( 3 )
# 	print( N.toStr( ) )
# 	print( N.predict( np.array(["b","3"]) ) )
# 	print( N.test( n ) )
# # endregion
