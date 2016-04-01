'''
this scripts get review and add label
'''

from os import listdir
import pandas as pd
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
import json
import os




#sub categories for automotive
sub_categories=['Auto Glass Services','Auto Parts & Supplies',
'Boat Dealers','Body Shops','Car Dealers','Car Wash','Gas & Service Stations',
'Oil Change Stations','Tires','Auto Detailing']


# Pull text data according to flag.csv
output_path='text/'

data_review=[]

for line in open('yelp_academic_dataset_review.json').readlines():
	data_review.append(json.loads(line))
df=pd.DataFrame(data_review)


df1 = pd.DataFrame.from_csv('multiclass.csv')
isTrain = {}
for index, row in df2.iterrows():
	isTrain[row['business_id']] = 1
keys = isTrain.keys()


prev_bId = "null"
fp = open(output_path+'test', "w")
for index, row in df.iterrows():
    bId = row['business_id']
    if bId in keys:
        if bId != prev_bId:
            fp.close()
            fp = open(output_path+bId, "w")

        review = row['text'] + "\n"

    	try:
      		fp.write(review)
      	except:
      		try:
      			fp.write(review.decode('utf-8'))
      		except:
      			print 'error'

        prev_bId = bId
fp.close()



# Get text from text/ and seperate them into yes and no

import pandas as pd
import numpy as np
import os

categories=['Home Services','Automotive','Pets','Education','Professional Services',
'Financial Services','Public Services & Government','Health & Medical','Restaurants']

for cat in categories:
  df1 = pd.DataFrame.from_csv(cat+'.csv',index_col=3)

  for index, row in df1.iterrows():

    bId = row['business_id']
    if row['Y'] == 1 :
      if not os.path.exists(cat+'/yes'+"/"+bId+"_1") :
        print '1'
        fp_yes = open(cat+'/yes'+"/"+bId+"_1", "w")
        for l in open('text/'+bId).readlines():
          fp_yes.write(l+' ')
        fp_yes.close()

    else: 
      if not os.path.exists(cat+'/no'+"/"+bId+"_0") :
        fp_no = open(cat+'/no'+"/"+bId+"_0", "w")
        for l in open('text/'+bId).readlines():
          fp_no.write(l+' ')
        fp_no.close()










