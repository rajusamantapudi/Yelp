import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_review=[]
for line in open('yelp_academic_dataset_review.json').readlines():
	data_review.append(json.loads(line))

data_review=pd.DataFrame(data_review)[['business_id','stars']]


data=[]
for line in open('yelp_academic_dataset_business.json').readlines():
	data.append(json.loads(line))

data=pd.DataFrame(data)[['business_id','categories','name','review_count','stars']]


#1: histogram /table of no of reviews/ stars 
#split multi categories
data_new = data.groupby('business_id').categories.apply(lambda x: pd.DataFrame(x.values[0])).reset_index().drop('level_1', axis = 1)
data_new.columns = ['business_id','categories']   

catdata=pd.merge(data_new, data.drop('categories',axis=1), on='business_id',how='left').groupby('categories')
catdata_stats=pd.concat([catdata['review_count'].sum(),catdata['stars'].agg([np.mean,np.std])],axis=1)



#summary of review count
catdata_stats['review_count'].describe()
# review count between (10,2500) is 70%
criterion = catdata_stats['review_count'].map(lambda x :x<2500 and x>10)
float(len(catdata_stats[criterion]))/len(catdata_stats)  # 0.7011494252873564

#histgram is bad
plt.xlabel('Number of reviews')
plt.ylabel('Frequncy')
plt.title(r'Histogram of Number of reviews')
plt.hist(catdata_stats['review_count'],bins=100)
plt.show()
#a partcial histgram
plt.xlabel('Number of reviews ranging from 10 to 2500 ')
plt.ylabel('Frequncy')
plt.title(r'Histogram of Number of reviews')
plt.hist(catdata_stats[criterion]['review_count'],bins=100)
plt.show()


#summary of ratings
catdata_stats['mean'].describe()
plt.xlabel('Average ratings')
plt.ylabel('Frequncy')
plt.title(r'Histogram of Average ratings')
plt.hist(catdata_stats['mean'])
plt.show()



#2 summary of multi-catogries:pie chart
number_of_categories=data_new.groupby('business_id').count()['categories'].reset_index().groupby('categories').count()

plt.title('Summary of multi-catogries')
labels=number_of_categories.index
# These are the "Tableau 20" colors as RGB.
tableau20 = [(111,54,98),(225,113,130),(225,174,93), (216,191,216), (248,222,189),
			(159,97,100),(208,152,238), 
			(148,103,189),(176,196,222),(247,182,210),(3,41,81)]
 
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

explode = np.zeros(10).map(lambda x:x+0.05)

plt.pie(number_of_categories,labels=labels,colors=tableau20, autopct='%1.1f%%',explode=explode)
plt.show()



#3: top 3000 business unit (most review) and their categories
#why i set 3000 biz units: to match the #2 case where the number of biz unites in largest catogies is 2000+
#Others need to be combined to others
cat_series=data.sort(columns='review_count',ascending=False)[0:3000].groupby('business_id').categories.apply(lambda x:pd.DataFrame(x.values[0])).reset_index().drop('level_1', axis = 1)
cat_series.columns = ['business_id','categories'] 
myData2=cat_series.groupby('categories').count().sort(columns='business_id',ascending=False) 
myData2[myData2['business_id']>200] 
myData2[myData2['business_id']>200] 

#4: business unit (9000+) which only has 3 reviews(the least) and their categories
#
cat_series=data[data['review_count']==3].groupby('business_id').categories.apply(lambda x:pd.DataFrame(x.values[0])).reset_index().drop('level_1', axis = 1)
cat_series.columns = ['business_id','categories'] 
myData=cat_series.groupby('categories').count().sort(columns='business_id',ascending=False) 
myData[myData['business_id']>200]      #this choice thas 22  categories












