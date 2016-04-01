'''
run trian.py to train the model
'''

import pandas as pd
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
import json
import os

# M1-> only BOH
# M2-> only name
# M3-> only review
# M4-> only nmf
# M5-> only lda
# M6-> BOH+name
# M7-> BOH+name+review
# M8-> BOK+name+review+lda
# M9-> BOH+nmae+review+nmf
# M10 -> Test on multi class

categories=['Active Life','Home Services','Automotive','Pets','Education','Professional Services',
'Financial Services','Public Services & Government','Health & Medical','Restaurants']

#sub categories for automotive
sub_categories=['Auto Glass Services','Auto Parts & Supplies',
'Body Shops','Car Dealers','Car Wash','Gas & Service Stations',
'Oil Change Stations','Tires','Auto Detailing']


def cleanDoc(doc):
        parts = doc.split('\n')
        newparts = []
        for part in parts:
        	try:
        		part=part.encode('utf-8').decode('utf-8','ignore').encode("utf-8")
        		newparts.append(part)
        		part.replace('\n',' ')
        	except UnicodeDecodeError:
        		part= ''

        return ' '.join(newparts)
	


def get_test_result(cat, clf, X_train, y_train, X_test,y_test, x,f,y_names=None, confusion=False):	
	clf.fit(X_train, y_train)
	y_predicted = clf.predict(X_test)
	n_classes = 2
	n_train=len(y_train)
	f.write('Category: '+cat+'\n')
	f.write('Summary: train sample size %d \n' %(n_train))
	f.write(sklearn.metrics.classification_report(y_test, y_predicted))
	arr=sklearn.metrics.confusion_matrix(y_test, y_predicted, labels=range(n_classes))
	print arr
	f.write(str(arr[0][0]))
	f.write(' ')
	f.write(str(arr[0][1]))
	f.write('\n')
	f.write(str(arr[1][0]))
	f.write(' ')
	f.write(str(arr[1][1]))
	f.write('\n')



def get_predict(cat, clf, X_train, y_train, X_test, bid):
	clf.fit(X_train, y_train)
	y_predicted = clf.predict(X_test)
	# Output predict result
	Output=open('predict_'+cat+'.csv','w')
	Output.write('business_id,Y\n')

	for i in np.arange(len(X_test)):
		Output.write(bid.iloc[i]+',')
		Output.write(str(y_predicted[i]))
		Output.write('\n')
	Output.close()



def prep_features(cat, x, path='data_10/Automotive/sub_sample_with_flag/', text_path='data_10/Automotive/text/'):
	df=pd.read_csv(path+cat+'.csv')[['business_id','Y','Flag']].sort(columns='business_id')
	df2=pd.read_csv('business_name.csv').sort(columns='business_id') 

	#df_ida= pd.read_csv('ldanmffinal/'+cat+'_Ida.csv').sort(columns='business_id').iloc[:,0:11]
	#df_nmf=pd.read_csv('ldanmffinal/'+cat+'_nmf.csv').sort(columns='business_id').iloc[:,0:11] #.rename(columns=['n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','business_id'])
	#df=df.merge(df_ida,on='business_id')
	#df=df.merge(df_nmf,on='business_id') 

	# M10 -> Test on multi class
	if x=='10':
		df_test=pd.read_csv('multiclass.csv')
		df_test['Flag']=0
		# Merge ida and nmf
		df_ida_test= pd.read_csv('multiclass_Ida.csv').sort(columns='business_id').iloc[:,0:11]
		df_nmf_test=pd.read_csv('multiclass_nmf.csv').sort(columns='business_id').iloc[:,0:11] #.rename(columns=['n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','business_id'])
		df_test=df_test.merge(df_ida_test,on='business_id')
		df_test=df_test.merge(df_nmf_test,on='business_id')


		df=df[df['Flag']==1] 
		# Combine df and df_test : Need test on this
		df=pd.concat([df,df_test])

	# Get BOH
	#df3=pd.read_csv('BHO-feature2.csv')
	#df=df.merge(df3,on='business_id')

	# Get name
	#df=df.merge(df2,on='business_id')


	# Get text
	df['text']=''
	for i in np.arange(len(df)):
		bId=df.ix[i,'business_id']	
		try:	
			df.ix[i,'text']=cleanDoc(open(text_path+bId).read()) #Change here if change to sub  catogory
		except :
			print 'empty text file: no review'
			pass

	df=df[df['text']!='']  # Check There are null value

	#df=df[df['name']!='']		

	df.to_csv('temp',index=False)
	df=pd.read_csv('temp')

	# Seperate train and test
	train=df[df['Flag']== 1]
	test=df[df['Flag']==0]

	y_train=train['Y']
	y_test=test['Y']

	print train.shape
	# Get X_train
	X_train_text, X_test_text=word_count_transform(train['text'],test['text'])
	#X_train_name, X_test_name=word_count_transform(train['name'],test['name'])

	if x=='1':
		X_train=train.iloc[:,-171:-3]
		X_test=test.iloc[:,-171:-3]
		#print X_test.shape, X_test.columns
		

	elif x=='2':
		X_train=X_train_name
		X_test=X_test_name

	elif x=='3':
		X_train=X_train_text
		X_test=X_test_text

	elif x=='4':
		X_train=train.iloc[:,13:23]
		X_test=test.iloc[:,13:23]

	elif x=='5':
		X_train=train.iloc[:,3:13]
		X_test=test.iloc[:,3:13]

	elif x=='6':  #M6 -> BOH+review
		X_train=np.concatenate([X_train_text,train.iloc[:,-171:-3]],axis=1) 
		X_test=np.concatenate([X_test_text,test.iloc[:,-171:-3]],axis=1) #		

	elif x=='7': # M7-> name+review
		X_train=np.concatenate([X_train_text,X_train_name],axis=1) #
		X_test=np.concatenate([X_test_text,X_test_name],axis=1) #

	elif x=='8' : # M8-> review+lda
		X_train=np.concatenate([X_train_text,train.iloc[:,3:13]],axis=1) #
		X_test=np.concatenate([X_test_text,test.iloc[:,3:13]],axis=1) #

	elif x=='10' :  # select a best model :name+review+lda
		X_train=np.concatenate([X_train_text],axis=1) #
		X_test=np.concatenate([X_test_text],axis=1) #

	else : # M9-> review+nmf
		X_train=np.concatenate([X_train_text,train.iloc[:,13:23]],axis=1) #
		X_test=np.concatenate([X_test_text,test.iloc[:,13:23]],axis=1) #	

	return X_train, X_test, y_train,y_test,test['business_id']



def word_count_transform(X_train, X_test):
	from sklearn.feature_extraction.text import TfidfVectorizer
	tfidf_vectorizer = TfidfVectorizer(stop_words='english')
	tfidf_vectorizer.fit(X_train)
	X_train_tfidf = tfidf_vectorizer.transform(X_train).todense()
	X_test_tfidf = tfidf_vectorizer.transform(X_test).todense()
	return X_train_tfidf, X_test_tfidf



def load_review (cat,path='sample_with_flag_2/'):
	# Creat file path
	Output_path=cat+'/'
	try:
		 os.mkdir(Output_path) 
		 os.mkdir(Output_path+'yes') 
		 os.mkdir(Output_path+'no') 
	except:
		print 'error'

	# Get sample file 
	df=pd.read_csv(path+cat+'.csv',index_col=0)

	# Get review file
	data_review=[]

	for line in open('yelp_academic_dataset_review.json').readlines():
		data_review.append(json.loads(line))
		df2=pd.DataFrame(data_review)

	print 'review load done'

	for index, row in df2.iterrows():
		bId = row['business_id']
		rId = row['review_id']

	# Get the flag of the bid
		if df.ix[bId,'Y']==1:
		    fp_yes = open(cat+'/yes'+"/"+rId, "w")
		    fp_yes.write(row['text'])
		    fp_yes.close()

		else: 
			fp_no = open(cat+'/no'+"/"+rId, "w")
			fp_no.write(row['text'])
			fp_no.close()



def plot_learning_curve(cat,ylim=None, cv=None, n_jobs=1):
	from sklearn.learning_curve import learning_curve
	import matplotlib.pyplot as plt
	import sklearn.svm
	import sklearn.metrics
	import sklearn.datasets

	# Load data 

	files = sklearn.datasets.load_files(cat+'/')
		
	for i, doc in zip(range(len(files.data)),files.data):
		files.data[i] = cleanDoc(doc)


	count_vector = sklearn.feature_extraction.text.CountVectorizer()
	word_counts = count_vector.fit_transform(files.data)

	tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
	X = tf_transformer.transform(word_counts)
	plt.figure()

	plt.title(cat)

	plt.xlabel("Training examples")
	plt.ylabel("Score")


	clf= sklearn.neighbors.KNeighborsClassifier()
	clf2 = sklearn.svm.LinearSVC() 

	train_sizes=[0.01,0.1,0.2,0.5,1]
	plt.figure()

	plt.title(cat)
	plt.xlabel("Training examples")
	plt.ylabel(" F1 Score")

	train_sizes, train_scores, test_scores = learning_curve(clf, X, files.target, scoring='f1', cv=5, train_sizes=[0.08,0.1,0.2,0.4,0.6,1])
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)			

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="y")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="b")
	plt.plot(train_sizes, train_scores_mean, 'o', color="y",label="Training score: KNN")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="b",label="Cross-validation score: KNN")
	print 'done knn'


	train_sizes, train_scores, test_scores = learning_curve(clf2 , X, files.target, scoring='f1', cv=5, train_sizes=[0.08,0.1,0.2,0.4,0.6,1])
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)


	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o', color="r",label="Training score: SVM")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score: SVM")

	print 'done svm'

	plt.legend(loc="best")
	plt.savefig(cat+'.png')
	return plt

plot_learning_curve('Automotive')

def plot(cat,clf,clf2,X_train,X_test,y_train,y_test):  # Plot for main category
	from scipy import sparse
	# plot learning curve
	from sklearn.learning_curve import learning_curve
	import matplotlib.pyplot as plt
	import sklearn.svm
	import sklearn.metrics
	#plot
	title = cat
	
	train_sizes=[0.01,0.1,0.2,0.5,1]
	plt.figure()

	plt.title(cat)
	
	plt.xlabel("Training examples")
	plt.ylabel(" F1 Score")

	train_sizes, train_scores, test_scores = learning_curve(clf, sparse.csr_matrix(np.concatenate([X_train, X_test])),np.concatenate([y_train,y_test]), scoring='f1', cv=5, train_sizes=[0.08,0.1,0.2,0.4,0.6,1])
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)			

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="y")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="b")
	plt.plot(train_sizes, train_scores_mean, 'o', color="y",label="Training score: KNN")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="b",label="Cross-validation score: KNN")
	print 'done knn'


	train_sizes, train_scores, test_scores = learning_curve(clf2, sparse.csr_matrix(np.concatenate([X_train, X_test])),np.concatenate([y_train,y_test]), scoring='f1', cv=5, train_sizes=[0.08,0.1,0.2,0.4,0.6,1])
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)


	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1,color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o', color="r",label="Training score: SVM")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score: SVM")

	print 'done svm'
	mini=min(test_scores_mean)-0.1
	ylim=(mini, 1.01)

	plt.ylim(*ylim)
	plt.grid()
	plt.legend(loc='lower right',prop={'size':10})

	plt.savefig(cat+'_learning_curve.png')



def find_best_k(cat,X_train, X_test, y_train,y_test,score):
	from sklearn.cross_validation import KFold
	x=np.concatenate([X_train, X_test])
	y=np.concatenate([y_train,y_test])
	print score
	score[cat]={}
	n=len(x) 
	for k in np.arange(1,11): 
		score[cat][k]=[]
	    #split dataset into k folds  
		for train_ix,test_ix in KFold(n,5):  
			X_test=x[test_ix,:]
			y_test=y[test_ix]
			X_train=x[train_ix,:]            
			y_train=y[train_ix]
			#print X_test.shape, y_test.shape, X_train.shape, y_train.shape 
			#modeling
			clf=sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
			clf.fit(X_train,y_train)
			f1_score = sklearn.metrics.f1_score(y_test,clf.predict(X_test))  #which function should i use?????????
			print f1_score
			#add auc into dict
			score[cat][k].append(f1_score)
	return score


def main():

	clf= sklearn.neighbors.KNeighborsClassifier()
	clf2 = sklearn.svm.LinearSVC()  
	#clf = sklearn.naive_bayes.MultinomialNB()
	for x in ['3']:
		clf = sklearn.svm.LinearSVC()  
		f=open('Report_'+x+'.txt','w')
		for cat in sub_categories:
			X_train, X_test, y_train,y_test,BID=prep_features(cat,x)
			print 'Feature extraction done'
			if x== '10':
				get_predict(cat, clf, X_train, y_train, X_test,BID)
			else:
			score={}
			score=find_best_k(cat,X_train, X_test, y_train,y_test,score)
			df=pd.DataFrame(score)
			df.to_csv('find_best_k.csv')
			print score

				get_test_result(cat, clf, X_train, y_train, X_test,y_test, x,f)

				plt=plot(cat,clf,clf2,X_train,X_test,y_train,y_test)

		f.close()


if __name__ == '__main__':
	main()

