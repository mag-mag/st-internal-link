import streamlit as st
import pandas as pd
from collections import Counter
from datetime import datetime



uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    all_df = pd.read_csv(uploaded_file)
    st.dataframe(all_df)

stop_words = pd.read_csv('stopword_farsi.csv')['token'].to_list()

file_col = all_df.columns.to_list() 
keyword_col = st.selectbox(label='Choose KEYWORD column name:',options=file_col)
title_col = st.selectbox(label='Choose TITLE column name:',options=file_col)
cluster_col = st.selectbox(label='Choose CLUSTER column name:',options=file_col)

def generate_N_grams(text,ngram):
  words=[word for word in text.split(" ") if word not in set(stop_words)]  
  temp=zip(*[words[i:] for i in range(0,ngram)])
  ans=[' '.join(ngram) for ngram in temp]
  return ans

if st.checkbox("Are you ready to generate internal links?"):

    

    cluster_name = all_df[cluster_col].unique()


    unigram_df = pd.DataFrame(columns=['token','count','cluster'])
    bigram_df = pd.DataFrame(columns=['token','count','cluster'])
    for index,cluster in enumerate(cluster_name[:200]):
        title_list = all_df[all_df[cluster_col]==cluster][title_col].to_list()
        title_str = " ".join(title_list)
        one_token = generate_N_grams(title_str,ngram=1)
        two_token = generate_N_grams(title_str,ngram=2)
        data_two = Counter(two_token)
        data_one = Counter(one_token)
        temp_one = pd.DataFrame(data_one.items(),columns=['token','count'])
        temp_one = temp_one[~temp_one['token'].isin(stop_words)]
        temp_one = temp_one.sort_values(by='count',ascending=False)[:30]
        temp_one['cluster'] = cluster
        unigram_df = pd.concat([unigram_df,temp_one])


        temp_two = pd.DataFrame(data_two.items(),columns=['token','count'])
        temp_two = temp_two[~temp_two['token'].isin(stop_words)]
        temp_two = temp_two.sort_values(by='count',ascending=False)[:30]
        temp_two['cluster'] = cluster
        bigram_df = pd.concat([bigram_df,temp_two])


    bigram_df = bigram_df.drop(columns=['count'])
    merge_df = pd.merge(bigram_df,bigram_df,on='token',how='inner')
    grouped_df = merge_df.groupby(by=['cluster_x','cluster_y']).count()
    grouped_df = grouped_df[grouped_df['token']!=30]
    grouped_df = grouped_df[grouped_df['token']>7].reset_index()
    
    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(grouped_df)
    today = datetime.today()
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f'internal-link-{str(today)}.csv',
        mime='text/csv',
    )
