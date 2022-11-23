import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf_vectorizer = TfidfVectorizer()
category_data = pd.read_csv("./data/major_category.csv")
def category(major_list):
  best_cosine = -1
  best_idx = -1
  for idx, x in enumerate(category_data['학과']):
        sentence= (major_list,x )
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentence)
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        #print(cosine_sim,category_data['중분류'][idx] )
        if best_cosine < cosine_sim:
          best_cosine = cosine_sim
          best_idx = idx
  return category_data['중분류'][best_idx]