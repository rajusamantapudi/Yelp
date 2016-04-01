import json
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import os

data_review=[]

for line in open('yelp_academic_dataset_review.json').readlines():
        data_review.append(json.loads(line))

data_review=pd.DataFrame(data_review)[['business_id','text']]

gb=data_review.groupby('business_id')


for f in os.listdir("Data Set")[1:]:
	if(f.startswith('Restaurant')):
		train=pd.read_csv("Data Set/"+f) 

		reviews=''
		for i in np.arange(len(train)):
			try:
				for r in gb.get_group(train.iloc[i,1])['text'].values:
					reviews=reviews+r.encode('utf-8')+'\n'
			except:
				pass


		# Generate a word cloud image
		# Display the generated image:
		# the matplotlib way:
		# take relative word frequencies into account, lower max_font_size
		wordcloud =  WordCloud(
		                          stopwords=STOPWORDS,
		                          background_color='white',
		                          width=1200,
		                          height=1000
		                         ).generate(reviews)
		import matplotlib.pyplot as plt

		plt.figure()
		plt.imshow(wordcloud)
		plt.axis("off")

		plt.savefig('Data Set/'+f[:-4]+'.png')
		plt.show()


