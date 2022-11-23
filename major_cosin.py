#!pip install konlpy
from konlpy.tag import Okt
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#전공데이터셋을 받아옵니다
data = pd.read_csv("./data/major_curri.csv")

okt = Okt()

def tokenizer(raw_texts, pos=["Noun","Alpha","Verb","Number"], stopword=[]):
    p = okt.pos(raw_texts, 
            norm=True,   # 정규화(normalization)
            stem=True    # 어간추출(stemming)
            )
    o = [word for word, tag in p if len(word) > 1 and tag in pos and word not in stopword]
    return(o)


tfidf_vectorize =TfidfVectorizer(
        tokenizer=tokenizer,  # 문장에 대한 tokenizer (위에 정의한 함수 이용)
        min_df=1,  # 단어가 출현하는 최소 문서의 개수
        sublinear_tf=True  # tf값에 1+log(tf)를 적용하여 tf값이 무한정 커지는 것을 막음
    )

tfidf_matrix = tfidf_vectorize.fit_transform(data['커리큘럼'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#기존 데이터프레임으로부터 전공의 이름을 key, 전공의 인덱스를 value로 하는 딕셔너리 title_to_index를 만들어둡니다.
major_index = dict(zip(data['학부/학과명'], data.index))
#기존 데이터프레임으로부터 단대의 이름을 key, 단대의 인덱스를 value로 하는 딕셔너리 title_to_index를 만들어둡니다.
college_index = dict(zip(data['단대명'], data.index))

def get_recommendations(major, cosine_sim=cosine_sim):
    # 선택한 전공명으로부터 해당 전공의 인덱스를 받아온다.
    if '대학' in major:
        idx = college_index[major]
    else:
        idx = major_index[major]

    # 해당 전공과 모든 전공과의 유사도를 가져온다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 전공들을 정렬한다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 전공들을 받아온다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 전공의 인덱스를 얻는다.
    major_indices = [idx[0] for idx in sim_scores]
    temp = data['학부/학과명'].iloc[major_indices]
    result = temp.to_json(force_ascii=False, orient='records')
    parsed = json.loads(result)
    result = ','.join(parsed)

    # 가장 유사한 10개의 전공의 이름을 리턴한다.
    return result

