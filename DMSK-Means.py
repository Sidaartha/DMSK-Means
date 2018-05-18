# ==================================== Importing Libraries ================================================

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# ===================================== Reading data file =================================================

iris=pd.read_csv("iris.data")
df = pd.DataFrame(iris)

Y=np.array(iris)
Y=np.delete(Y,np.s_[0:4],1)
for i in range(len(Y)):
	if Y[i] == 'Iris-setosa':
		Y[i]=0
	elif Y[i] == 'Iris-versicolor':
		Y[i]=1
	elif Y[i] == 'Iris-virginica':
		Y[i]=2
	else:
		pass
Y=np.reshape(Y,len(Y))
X=np.array(iris)
X=np.delete(X,4,1)

# ===================================== Defining Functions =================================================

#Random centroid initiation
def Centroid_init(K):

	centroid=[]
	for i in range(K):
		centroid.append(random.choice(X))
	centroid=np.array(centroid)

	return centroid

#Labeling data based on centroid
def Labeling(centroid):

	label=[]
	label_bor=[]
	for i in range(len(X)):
		A = X[i]
		distance=[]
		for e in range(len(centroid)):
			B = centroid[e]
			D=np.sqrt(np.sum(A-B)**2)
			distance.append(D)
		idx = np.argmin(distance)
		idx_max = np.argmax(distance)
		label.append(idx)
		label_bor.append(idx_max)

	return label, label_bor

#New centroid from labeled data i.e, Avg of labeled.pt
def Centroid_change(label, K_cent):

	new_cent=[]
	for i in range(K_cent):
		new_cent_i=[]
		for e in range(len(label)):
			if label[e] == i:
				nc = label[e]
				new_cent_i.append(X[e])
			else :
				pass
		new_cent_i=np.array(new_cent_i)
		if len(new_cent_i)==0:
			pass
		else:
			new_cent.append(new_cent_i.sum(axis=0)/len(new_cent_i))
	new_cent=np.array(new_cent)

	return new_cent

#Accurace function 
def Accuracy(Final_label, Y):

	count=0
	for i in range(len(Y)):
		if Y[i]==Final_label[i]:
			count=count+1
		else:
			pass
	accuracy=count/len(Y)

	return accuracy, count

#KMeans function which itterates untill there is no change in centroid
def KMeans():

	Initial_cen=Centroid_init(K)
	Cent_Store=[]
	count=0
	while True:
		if count == 0:
			Label_labeling, _ = Labeling(Initial_cen)
		else :
			Label_labeling, _ = Labeling(Changed_centroid)
		Changed_centroid = Centroid_change(Label_labeling, K)
		Cent_Store.append(Changed_centroid)
		if count == 0:
			pass
		else:	
			count_p=count-1
			if np.all(Cent_Store[count_p]==Cent_Store[count]):
				break
			else:
				pass
		count=count+1
	Final_labeling, Final_labeling_bor=Labeling(Changed_centroid)
	Final_labeling=np.array(Final_labeling)

	return Final_labeling, Final_labeling_bor, Changed_centroid, count

#Centroid merging if distance b/w two centroids is less than min_dis
def Centroid_merge(Cent_to_merge):

	Cent_dis=[]
	Cent_i=[]
	Cent_e=[]
	Final_Centroid_merge=Cent_to_merge
	c=0
	for i in range(len(Final_Centroid_merge)):
		count=0
		for e in range(len(Final_Centroid_merge)):
			if e > i:
				dis=np.sqrt(np.sum(Final_Centroid_merge[i]-Final_Centroid_merge[e])**2)
				Cent_dis.append(dis)
				Cent_i.append(i)
				Cent_e.append(e)
				# print(dis)
			else:
				pass
	for i in range(len(Cent_dis)):
		if Cent_dis[i]<min_dis:
			Cent_merge=sum(Final_Centroid_merge[Cent_i[i]] , Final_Centroid_merge[Cent_e[i]])/2
			Final_Centroid_merge=np.delete(Final_Centroid_merge, (Cent_i[i],Cent_e[i]), 0)
			Final_Centroid_merge=np.insert(Final_Centroid_merge, Cent_i[i], Cent_merge, 0)
			Final_Centroid_merge=np.insert(Final_Centroid_merge, Cent_e[i], 'NA', 0)
		else:
			pass
	merge_id=0
	while True:
		if len(Final_Centroid_merge)<=merge_id:
			break
		elif Final_Centroid_merge[merge_id].all()=='NA':
			Final_Centroid_merge=np.delete(Final_Centroid_merge, merge_id, 0)
		else :
			pass
		merge_id=merge_id+1

	return Final_Centroid_merge

#Centroid split if radius of a centroids is grater than max_rad with distant.pt as new centroid
def Centroid_split(Cent_a_merge, label_merge):

	Dis_max=[]
	Dis_max_id_X=[]
	for i in range(len(Cent_a_merge)):
		Distance_max=[]
		Distance_max_id_X=[]
		for e in range(len(X)):
			if label_merge[e]==i:
				distance_max=np.sqrt(np.sum(Cent_a_merge[i]-X[e])**2)
				Distance_max.append(distance_max)
				Distance_max_id_X.append(e)
			else:
				pass
		if not Distance_max:
			max_dis_cent=0
			id_max_X='NA'
			Dis_max.append(max_dis_cent)
			Dis_max_id_X.append(id_max_X)
			pass
		else:
			id_max_dis=np.argmax(Distance_max)
			max_dis_cent=Distance_max[id_max_dis]
			id_max_X=Distance_max_id_X[id_max_dis]
			Dis_max.append(max_dis_cent)
			Dis_max_id_X.append(id_max_X)
	Centroid_split_only=[]
	for i in range(len(Cent_a_merge)):
		if Dis_max[i]>max_rad:
			Centroid_split_only.append(X[Dis_max_id_X[i]])		
		else:
			pass
	Centroid_split_only=np.array(Centroid_split_only)
	if len(Centroid_split_only)==0:
		Final_Centroid_split=Cent_a_merge
	else:
		Final_Centroid_split=np.concatenate((Cent_a_merge, Centroid_split_only))

	return Final_Centroid_split

# ==================================== Initial KMeans operation =========================================

K=3
itt=30
Final_label_itt=[]
Final_label_bor_itt=[]
Final_Centroid_itt=[]
Itteration_itt=[]
Accuracy_itt=[]
Count_iit=[]

#Itterates untill reaches global optima as KMeans may converges at local optima based on initial centroid
for i in range(itt):
	Final_label, Final_label_bor, Final_Centroid, Itteration=KMeans()
	Acc, cou =Accuracy(Final_label, Y)
	Final_label_itt.append(Final_label)
	Final_label_bor_itt.append(Final_label_bor)
	Final_Centroid_itt.append(Final_Centroid)
	Itteration_itt.append(Itteration)
	Accuracy_itt.append(Acc)
	Count_iit.append(cou)
Max_id=np.argmax(Accuracy_itt)
print(Accuracy_itt[Max_id])
print(len(X)-Count_iit[Max_id])

# ========================= Distance based Splitting & Merging through KMeans ============================

min_dis=4
max_rad=3.5
count_DSM=0
itt_count=0
Cent_to_merge=[]
SM_Cent=[]

while True:

	if count_DSM == 0:
		Final_Centroid_merge = Centroid_merge(Final_Centroid_itt[Max_id])
		# print(count_DSM)
		# print('Final_Centroid_merge')
		# print(Final_Centroid_merge)
	else :
		Final_Centroid_merge = Centroid_merge(Cent_to_merge)
		# print(count_DSM)
		# print('Final_Centroid_merge')
		# print(Final_Centroid_merge)
	Label_merge, _ =Labeling(Final_Centroid_merge)
	New_Final_Centroid_merge=Centroid_change(Label_merge, len(Final_Centroid_merge))
	# print('New_Final_Centroid_merge')
	# print(New_Final_Centroid_merge)
	Final_Centroid_split = Centroid_split(New_Final_Centroid_merge, Label_merge)
	# print('Final_Centroid_split')
	# print(Final_Centroid_split)
	Label_split, _ =Labeling(Final_Centroid_split)
	New_Final_Centroid_split=Centroid_change(Label_split, len(Final_Centroid_split))
	# print('New_Final_Centroid_split')
	# print(New_Final_Centroid_split)
	Cent_to_merge = New_Final_Centroid_split
	count_DSM=count_DSM+1
	SM_Cent.append(Cent_to_merge)
	SM_Cent_arr=np.array(SM_Cent)
	# print('----------------------')
	# print(count_DSM)
	# print(SM_Cent_arr[-1])
	if itt_count==5000:
		print('Progressing....')
		print(count_DSM-1)
		# print(SM_Cent_arr[-1])
		itt_count=0
	else:
		pass
	if count_DSM<3:
		pass
	elif np.all(SM_Cent_arr[-1]==SM_Cent_arr[-2]):
		print('Converged')
		print(SM_Cent_arr[-1])
		print(count_DSM)
	else:
		pass
	itt_count+=1

# =======================================================================================================

