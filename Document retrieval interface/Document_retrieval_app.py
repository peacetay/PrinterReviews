import streamlit as st
import re
import pandas as pd
from gensim import similarities
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from nltk.corpus import stopwords
from nltk.stem.porter import *
import gensim
from gensim import corpora
from gensim import models
import joblib
from nltk.stem.porter import PorterStemmer

###############################################################################################################################################
# my_expander_dkf = st.expander(label='Document retrieval')
# with my_expander_dkf:


st.header("Find related reviews")
# st.write("Select a topic to find related reviews.")

cols = st.columns(2)
Brand = ['HP', 'Canon', 'Epson']
brand = cols[0].multiselect(
        label="Select Brand",
        options=Brand
    )
Rating = [1, 2, 3, 4, 5]
rating= cols[1].multiselect(
        label="Select Rating",
        options=Rating
    )

Topic = ["Printing job",
     "Support and returns",
    "Print and scan quality",
     "Setup and connection",
    "Cartridge replacement",
    ""]

topic = st.selectbox(label="Select Topic",
        options=Topic)


# with st.expander('Type your own keywords'):
input_keywords = st.text_input('Type Keywords')

start_price, end_price = st.select_slider(
    'Price range',
    options=[70.0,100.0,130.0,160.0,190.0,210.0,240.0,290.0,350.0,400.0,500.0],
    value=(70,500))

 # Import data
@st.cache_data
def load_reviews():
    data_path = r'processed_data.joblib'
    df = joblib.load(data_path)
    return df

df = load_reviews()

hp_model_list = sorted([text for text in df['Review Model'].unique().tolist() if 'HP' in text])
comp_model_list = sorted([text for text in df['Review Model'].unique().tolist() if 'HP' not in text])

apply_button = st.button('Apply')

if apply_button:
    # input_text = key_words if key_words != '' else input_keywords
    input_text = input_keywords
   
    # compute similarity score from result of TFIDF Bigram
    @st.cache_data
    def compute_similarity():
        dictionary_bi = corpora.Dictionary(df['Processed_bigram'])
        corpus_bi = [dictionary_bi.doc2bow(text) for text in df['Processed_bigram']]
        TFIDF_bi = models.TfidfModel(corpus_bi)
        corpus_TFIDF_bi = [TFIDF_bi[vec] for vec in corpus_bi]
        IndexTFIDF_bi = similarities.SparseMatrixSimilarity(corpus_TFIDF_bi, len(dictionary_bi))
        return dictionary_bi, TFIDF_bi,IndexTFIDF_bi

    dictionary_bi, TFIDF_bi,IndexTFIDF_bi = compute_similarity()
    # Process query
    @st.cache_data
    def query(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            unigrams = word_tokenize(text)
            bigrams_list = list(bigrams(unigrams))

            # Stemming
            stemmer = PorterStemmer()
            stemmed_unigrams = [stemmer.stem(word) for word in unigrams]
            stemmed_bigrams = [tuple(stemmer.stem(word) for word in bigram) for bigram in bigrams_list]

            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            unigrams_without_stopwords = [word for word in stemmed_unigrams if word not in stop_words]
            bigrams_without_stopwords = [' '.join(bigram) for bigram in stemmed_bigrams if not any(word in stop_words for word in bigram)]

            # Join both unigrams and bigrams
            tokens = unigrams_without_stopwords + bigrams_without_stopwords 
            
            qVector_bi = dictionary_bi.doc2bow(tokens)
            qVectorTFIDF_bi = TFIDF_bi[qVector_bi]
            return qVectorTFIDF_bi
    
    rating_list = []
    for i in rating:
        rating_list.append(i)
    
    brand_list = []
    for i in brand:
        brand_list.append(i)


    if input_text:    
        qVectorTFIDF_bi = query(input_text)
        simTFIDF_bi = IndexTFIDF_bi[qVectorTFIDF_bi]
        df['Cosine similarity'] = simTFIDF_bi
        df_tfidf_bi = df.sort_values(by = 'Cosine similarity', ascending=False)
        selected_columns = ["Brand",  "Review rating", "Original title", "Original review","Review Model","List price", "Review date", "Cosine similarity", "Topic", "Probability"]
        # pd.set_option('display.max_colwidth', None)

        df_select = df_tfidf_bi[selected_columns] 
        if topic:
            if 'HP' in brand_list:
                df_final = df_select[(df_select['Brand']=='HP') & (df_select['Topic']==topic) & (df_select['Review rating'].isin(rating_list)) 
                                    & (df_select['List price']>=start_price) & (df_select['List price']<=end_price)
                                    & (df_select['Probability']>=0.3) & (df_select["Cosine similarity"]>=0.1)]
                df_final = df_final.drop_duplicates()
                df_final.sort_values(by=['Cosine similarity', 'Probability'], inplace=True, ascending = False)
                df_final = df_final.reset_index(drop=True)
                df_final.index += 1
                with st.expander('HP Review'):
                    st.dataframe(df_final)

            if 'Canon' in brand_list:
                df_canon = df_select[(df_select['Brand']=='Canon') & (df_select['Topic']==topic) & (df_select['Review rating'].isin(rating_list)) 
                                    & (df_select['List price']>=start_price) & (df_select['List price']<=end_price)
                                    & (df_select['Probability']>=0.3) & (df_select["Cosine similarity"]>=0.1)]
                df_canon = df_canon.drop_duplicates()
                df_canon.sort_values(by=['Cosine similarity', 'Probability'], inplace=True, ascending = False)
                df_canon = df_canon.reset_index(drop=True)
                df_canon.index += 1
                with st.expander('Canon Review'):
                    st.dataframe(df_canon)

            if 'Epson' in brand_list:
                df_comp = df_select[(df_select['Brand']== 'Epson') & (df_select['Topic']==topic) & (df_select['Review rating'].isin(rating_list)) 
                                    & (df_select['List price']>=start_price) & (df_select['List price']<=end_price)
                                    & (df_select['Probability']>=0.3) & (df_select["Cosine similarity"]>=0.1)]
                df_comp = df_comp.drop_duplicates()
                df_comp.sort_values(by=['Cosine similarity', 'Probability'], inplace=True, ascending = False)
                df_comp = df_comp.reset_index(drop=True)
                df_comp.index += 1
                with st.expander('Epson Review'):
                    st.dataframe(df_comp)
        else:
            if 'HP' in brand_list:
                df_final = df_select[(df_select['Brand']=='HP') & (df_select['Review rating'].isin(rating_list)) 
                                    & (df_select['List price']>=start_price) & (df_select['List price']<=end_price)
                                     & (df_select['Probability']>=0.3)& (df_select["Cosine similarity"]>=0.1)]
                df_final['Topic'] = df_final.groupby('Original review')['Topic'].transform(lambda x: ', '.join(x))
                df_final['Probability'] = df_final.groupby('Original review')['Probability'].transform(lambda x: ', '.join(x.astype(str)))
                df_final.sort_values(by=['Cosine similarity', 'Probability'], inplace=True, ascending = False)
                df_final = df_final.drop_duplicates()
                df_final = df_final.reset_index(drop=True)
                df_final.index += 1
                with st.expander('HP Review'):
                    st.dataframe(df_final)

            if 'Canon' in brand_list:
                df_canon = df_select[(df_select['Brand']=='Canon')  & (df_select['Review rating'].isin(rating_list)) 
                                    & (df_select['List price']>=start_price) & (df_select['List price']<=end_price)
                                      & (df_select['Probability']>=0.3)& (df_select["Cosine similarity"]>=0.1)]
                df_canon['Topic'] = df_canon.groupby('Original review')['Topic'].transform(lambda x: ', '.join(x))
                df_canon['Probability'] = df_canon.groupby('Original review')['Probability'].transform(lambda x: ', '.join(x.astype(str)))
                df_canon = df_canon.drop_duplicates()
                df_canon.sort_values(by=['Cosine similarity', 'Probability'], inplace=True, ascending = False)
                df_canon = df_canon.reset_index(drop=True)
                df_canon.index += 1
                with st.expander('Canon Review'):
                    st.dataframe(df_canon)

            if 'Epson' in brand_list:
                df_comp = df_select[(df_select['Brand']== 'Epson')  & (df_select['Review rating'].isin(rating_list)) 
                                    & (df_select['List price']>=start_price) & (df_select['List price']<=end_price)
                                     & (df_select['Probability']>=0.3)& (df_select["Cosine similarity"]>=0.1)]
                df_comp['Topic'] = df_comp.groupby('Original review')['Topic'].transform(lambda x: ', '.join(x))
                df_comp['Probability'] = df_comp.groupby('Original review')['Probability'].transform(lambda x: ', '.join(x.astype(str)))
                df_comp = df_comp.drop_duplicates()
                df_comp.sort_values(by=['Cosine similarity', 'Probability'], inplace=True, ascending = False)
                df_comp = df_comp.reset_index(drop=True)
                df_comp.index += 1
                with st.expander('Epson Review'):
                    st.dataframe(df_comp)