import numpy as np
import matplotlib.pyplot as plt
from math import *
from operator import itemgetter

def mean_absolute_error(real,predicted):
	value = 0
	for i in range(len(real)):
		value += abs(float(real[i])-float(predicted[i]))
		
	value = value/len(real)
	return value
	
def root_mean_square_error(real,predicted):
	value = 0
	for i in range(len(real)):
		value += (float(real[i])-float(predicted[i]))**2
		
	value = (value/len(real))**0.5
	return value	

def data_real_predicted(filename,sep):
	real = []
	predicted = []
	reader = open(filename)
	for line in reader:
		line = line.strip()
		data = line.split(sep)
		
		real.append(data[2])
		predicted.append(data[3])
		
	reader.close()
	
	return real,predicted

def data_topn(filename,sep):
	real = []
	predicted = []
	active_user = 0
	flag = False
	order_r = []
	order_p = []
	reader = open(filename)
	for line in reader:
		line = line.strip()
		data = line.split(sep)

		if active_user!=data[0]:
			active_user = data[0]
			if flag:
				real.append(sorted(order_r,key=lambda x: x[1],reverse=True))
				predicted.append(sorted(order_p,key=lambda x: x[1],reverse=True))
			order_r = []
			order_p = []
		
		
		order_r.append(tuple([data[1],float(data[2])]))
		order_p.append(tuple([data[1],float(data[3])]))
		flag = True
		
		
	reader.close()
	
	return real,predicted

def discounted_cumulative_gain(ranking_list,p):
	dcg = 0
	for i in range(1,p+1):
		dcg += (2**(ranking_list[i][1])-1)/log(i+1,2)
	
	return dcg

def average_ndcg(list_r,list_p,p):
	all_values = []
	for i in range(p):
		all_values.append(discounted_cumulative_gain(list_p[i],p)/discounted_cumulative_gain(list_r[i],p))
	
	max_value = max(all_values)
	min_value = min(all_values)
	average = float(sum(all_values)/len(all_values))
	
	return max_value,min_value,average

def data_error(filename,sep):	
	X = []
	Y = []
	reader = open(filename)
	for line in reader:
		line = line.strip()
		data = line.split(sep)
		
		iteration = data[0]
		error_val = data[1]		
		
		X.append(data[0])
		Y.append(data[1])
	
	reader.close()
	return X,Y


def plot_error():
	x,y = data_error('GDM_1_100.txt',' ')
	plot1, = plt.plot(x,y)
	x,y = data_error('GDM_3_100.txt',' ')
	plot2, = plt.plot(x,y)
	x,y = data_error('GDM_10_100.txt',' ')
	plot3, = plt.plot(x,y)
	
	plt.grid(True)
	plt.title('Error function value with different latent factors')
	plt.xlabel('iterations')
	plt.ylabel('error')
	plt.legend([plot1,plot2,plot3],["1 latent factor", "3 latent factors","10 latent factors"])
	plt.show()


print "Without tags"
r,p = data_real_predicted('predicted1_wtag.txt','::')
mae = mean_absolute_error(r,p)
rmse = root_mean_square_error(r,p)

print "mae"
print mae
print "rmse"
print rmse

print "With tags"
r,p = data_real_predicted('predicted1_tag.txt','::')
mae = mean_absolute_error(r,p)
rmse = root_mean_square_error(r,p)

print "mae"
print mae
print "rmse"
print rmse


print "nDCG With tags"
for i in [1,3,5,10,20]:
	print i
	r,p = data_topn('predicted1_tag.txt','::')
	a,b,c = average_ndcg(r,p,i)
	print a,b,c

print "nDCG Without tags"
for i in [1,3,5,10,20]:
	print i
	r,p = data_topn('predicted1_wtag.txt','::')
	a,b,c = average_ndcg(r,p,i)
	print a,b,c


plot_error()

