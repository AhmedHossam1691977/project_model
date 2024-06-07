
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pymongo.mongo_client import MongoClient

def get_data():
    client = MongoClient("mongodb+srv://Admin:AhmedHossam1691977@cluster0.8zhjqva.mongodb.net/E-commerce")

    products = client["E-commerce"]["products"]

    data = list(products.find())

    df = pd.DataFrame(data)

    df[["_id","brand","title","slug","description","catigory","subCategory","createdAt","updatedAt"]]= df[["_id","brand","title","slug","description","catigory","subCategory","createdAt","updatedAt"]].astype(str)
    
    return df

def clean_data():
    df = get_data()

    df.drop(df[df["title"] == "hhhhhhhhhhhhhhhhhhhhh"].index,inplace=True)

    df.drop(df[df["title"] == "dsigdifuhdid"].index,inplace=True)

    df["sold"] = df["sold"].fillna(0)

    return df

def similarity():
    df = clean_data()

    df['combined_features'] = df.apply(lambda x: ' '.join([str(x['title']), str(x['description']), str(x['catigory']), str(x['brand'])]), axis=1)
    
    tfidf = TfidfVectorizer(stop_words='english')
    
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    return cosine_sim



def get_recommendations(title):
    df = clean_data()

    idx = df.index[df['title'] == title].tolist()[0]

    cosine_sim = similarity()

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]  

    product_indices = [i[0] for i in sim_scores]

    return df.iloc[product_indices].to_dict('records')
