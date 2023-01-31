import numpy as np
import sys, os
from decisionTree import DecisionTree as DT


# sys.argv[1:] = (lambda name:
# 		[f"./data/{name}_train.tsv", f"./out/{name}_inspect.tsv"]) \
# 	(["small", "politicians", "education", "mushroom"] [0])

if __name__ == "__main__":
	try: _, inPath, outPath, *_ = sys.argv
	except: print ("Usage: inspection.py <in:tsv> <out:txt>"); exit(1)
	categs = np.loadtxt (inPath, dtype=str, comments=None, delimiter="\t", converters=None, skiprows=1, usecols=-1, unpack=False)
	freqs = DT.countFreq(categs)[1]
	rslt = (DT.entropy(freqs), 1-np.max(freqs))
	os.makedirs (os.path.dirname (outPath), exist_ok=True)
	with open (outPath, "w") as outFile:
		outFile.write (f"entropy: {rslt[0]}\nerror: {rslt[1]}\n")
