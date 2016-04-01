'''
The class sample from '/data_10', which contains 10 primary categories business
units infomation.

Input: Category; sampleSize

Output: one dataset for that category classifier

'''

from os import listdir
import pandas as pd
from numpy import random
import numpy as np
import os

categories=['Active Life','Home Services','Automotive','Pets','Education','Professional Services',
'Financial Services','Public Services & Government','Health & Medical','Restaurants']
sub_categories=['Auto Glass Services','Auto Parts & Supplies',
'Boat Dealers','Body Shops','Car Dealers','Car Wash','Gas & Service Stations',
'Oil Change Stations','Tires','Auto Detailing']

dir_path='data/'


def clean(category):
	others=categories[:]
	others.remove(category)
	path='data_10/'+category+'.csv'

	result=[]
	for l in open(path).readlines():
		flag=1
		for i in others:
			if l.find(i)!=-1:
				print i,l
				flag=0
				break
		if flag==1:
			result.append(l)

	print len(result)
	output_file_name='data/'+category+'.csv'
	f=open(output_file_name,'w')
	for i in result:
		f.write(i+'\n')


# Sample 10% from population
def sample(category,p=0.1):
	file_name=dir_path+category+'.csv'
	df=pd.read_csv(file_name);
	df2=df.sample(frac=p)
	df2.to_csv('data/sample/'+category+'.csv',index=False)



# Combine all category data
# Make a sample for each category classifer
def make_sample(category):
	file_name=dir_path+'sample/'+category+'.csv'
	output_file_name='newsample/'+category+'.csv'
	dfs=[]
	df=pd.read_csv(file_name);
	df['Y']=1
	dfs.append(df)
	for j in categories:
		if (j!=category):
			df2=pd.read_csv(dir_path+'sample/'+j+'.csv');
			df2['Y']=0
			dfs.append(df2)
	
	result=pd.concat(dfs)
	result.to_csv(output_file_name,index=False)


def get_multi_sample(category):
	file_name='data_10/'+category+'.csv'

	result=[]
	for l in open(file_name).readlines():
		count=0
		for i in categories:
			if l.find(i)!=-1:
				count=count+1
		if count>1:
			result.append(l)

	output_file_name='multiclass/'+category+'.csv'
	f=open(output_file_name,'w')
	for i in result:
		f.write(i+'\n')



def devide_data(cat,sub_categories,path='data_10/'):
	try:
		os.mkdir(path+cat+'/')
	except:
		pass	
	for c in sub_categories:
		print c
		record=[]
		for l in open(path+cat+'.csv').readlines()[1:]:
			if l.find(c)!=-1:
				record.append(l[:-1]+',1')
			else:
				record.append(l[:-1]+',0')

		f=open(path+cat+'/'+c+'.csv','w')
		f.write(',business_id,categories,Y'+'\n')
		for r in record:
			f.write(r+'\n')
		f.close()
		print 'done one'


def append_flag(cat):
	path='newsample/'+cat+'.csv'
	flag=pd.read_csv('newsample/flag.csv')
	df=pd.read_csv(path)
	result=pd.merge(df,flag,on='business_id')
	result[['business_id','Y','Flag']].to_csv('newsample/sample_with_flag_2/'+cat+'.csv',index=False)




def combine(path):
	result=[]
	for cat in categories:
		subpath=path+cat+'.csv'
		for i in open(subpath).readlines()[:]:
			result.append(i)
	f=open('multiclass.csv','w')
	for i in result:
		f.write(i)




def main():

	for i in sub_categories:
		df=pd.read_csv('data_10/Automotive/sub_sample_with_flag/'+i+'.csv')
		print len(df), len(df.business_id.unique())

	dfs=[]
	for i in categories:
		df=pd.read_csv('multiclass/'+i+'.csv',index_col=0,names=['index','business_id','categories'])
		dfs.append(df)
	pd.concat(dfs,ignore_index=True).to_csv('multiclass.csv')


	df=pd.read_csv('multiclass.csv')

	df[df['business_id'].duplicated()==False].iloc[:,1:].to_csv('multiclass.csv',index=False)


	ids=open('train.csv').readlines()
	file_name='multiclass.csv'
	df=pd.read_csv(file_name)
	df['F']=1
	for index, row in df.iterrows():
		for j in ids:
			if row['business_id'].find(j)!=-1:
				print row['business_id'],j
				row['F']=0
	df.to_csv('nulticlass_new.csv')


	#deivde a primary dataset into several sub_categories
	#need to input sub_categories lists



	for i in categories:
		append_flag(i)	

	for cat in categories:
		df=pd.read_csv('newsample/sample_with_flag_2/'+cat+'.csv')
		print df[df['business_id'].duplicated()]




	# make a flag
	df=pd.read_csv('newsample/Active Life.csv')

	n=int(len(df)*0.8)

	flag=np.zeros(len(df))
	for i in np.arange(n):
		flag[i]=1

	random.shuffle(flag)
	for i in np.arange(len(df)):
		df.ix[i,'Flag']=flag[i]

	df[['business_id','Flag']].to_csv('newsample/flag.csv')



	os.mkdir('data_10/Automotive/WithFlag/')
	df=pd.read_csv('data_10/Automotive/Tires.csv')
	n=int(len(df)*0.8)
	flag=np.zeros(len(df))
	for i in np.arange(n):
		flag[i]=1

	random.shuffle(flag)
	for i in np.arange(len(df)):
		df.ix[i,'Flag']=flag[i]

	df[['business_id','Flag']].to_csv('data_10/Automotive/flag.csv')
	for i in sub_categories:
		append_flag(i)



	for cat in categories:
		flag=pd.read_csv('flag2.csv')
		path='sample/'+cat+'.csv'
		to_be_merged=pd.read_csv(path)
		result=pd.merge(flag,to_be_merged.iloc[:,1:],on='business_id',how='inner')

		#os.mkdir('sample_with_flag/')
		result.to_csv('sample_with_flag/'+cat+'.csv',index=False)

	 

	df=pd.read_csv('sample/Automotive.csv')

	n=int(len(df)*0.8)

	flag=np.zeros(len(df))
	for i in np.arange(n):
		flag[i]=1

	random.shuffle(flag)
	for i in np.arange(len(df)):
		df.ix[i,'Flag']=flag[i]

	df[['business_id','Flag']].to_csv('flag.csv')




if __name__ =='__main__':
	main()
