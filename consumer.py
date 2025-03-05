import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df.dropna()
    return df

def plot_rating_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Cons_rating', kde=True)
    plt.title('Rating Distribution')
    st.pyplot(plt)

def plot_class_distribution(df):
    clothing_counts = df['Cloth_class'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=clothing_counts.index, y=clothing_counts.values, alpha=0.8)
    plt.title('Clothing Class Distribution')
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.xticks(rotation=90)
    st.pyplot(plt)

def plot_wordcloud(df):
    word_could_dict = Counter(df['Cloth_class'].tolist())
    wordcloud = WordCloud(width=1000, height=800).generate_from_frequencies(word_could_dict)
    plt.figure(figsize=(16,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    st.pyplot(plt)

# Streamlit UI
st.title("Consumer Review Analysis")
st.write("Upload a CSV file containing clothing reviews and visualize insights.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    
    st.write("### Rating Distribution")
    plot_rating_distribution(df)
    
    st.write("### Clothing Class Distribution")
    plot_class_distribution(df)
    
    st.write("### WordCloud of Clothing Categories")
    plot_wordcloud(df)
