import csv
import numpy as np
from sklearn import tree

#Load Features Dataset
def loadCSV_features(filename_features):
	lines = csv.reader(open(filename_features, "rb"))
	dataset_features = list(lines)
	for i in range(len(dataset_features)):
		dataset_features[i] = [float(x) for x in dataset_features[i]]
	return dataset_features

#Load Label Dataset
def loadCSV_label(filename_label):
	lines = csv.reader(open(filename_label, "rb"))
	dataset_label = list(lines)
	for i in range(len(dataset_label)):
		dataset_label[i] = [float(x) for x in dataset_label[i]]
	return dataset_label

#Main Declaration
def main():
		filename_features = 'features.csv'
		features=loadCSV_features(filename_features)
		filename_label = 'label.csv'
		label = loadCSV_label(filename_label)
		X=features
		label=np.array(label)
		y=np.ravel(label)	
		clf=tree.DecisionTreeClassifier()
		clf=clf.fit(features,label)
		input=[[48,1,2,110,229,0,0,168,0,1,3,0,7]]
		out=clf.predict(input) #Pass the required Testing Data as a List
		result= out[0]
		print(result)

main()
		