# Consumer Review

## 📌 Overview
This project analyzes consumer reviews of clothing products using Python and data visualization techniques. The dataset is loaded from Google Drive, cleaned, and then explored through various statistical and visual analyses. The project focuses on:
- Data preprocessing (handling missing values)
- Rating distribution analysis
- Visualizing word frequency using WordCloud
- Exploratory Data Analysis (EDA) on different clothing categories

## 🔧 Technologies Used
- **Python** (for data analysis and visualization)
- **Pandas** (for data manipulation)
- **Matplotlib & Seaborn** (for data visualization)
- **WordCloud** (for text analysis)
- **NLTK & Gensim** (for natural language processing)
- **Latent Dirichlet Allocation (LDA)** (for topic modeling)
- **Streamlit** (for deployment)

## 📂 Dataset
The dataset used in this project is stored in Google Drive and loaded as a CSV file:
```
df_dataset_clothing = pd.read_csv("/content/drive/MyDrive/datvisalvin/data_amazon.xlsx - Sheet1.csv")
```

## 🚀 How to Run
1. Mount Google Drive:
```python
from google.colab import drive
drive.mount("/content/drive")
```
2. Install required libraries:
```bash
pip install streamlit pandas matplotlib seaborn wordcloud
```
3. Load and preprocess the dataset:
```python
df_dataset_clothing = df_dataset_clothing.dropna()
df_dataset_clothing.to_csv("/content/drive/MyDrive/datvisalvin/hasil_sortir_data_amazon.xlsx - Sheet1.csv")
```
4. Run data visualization:
```python
plt.figure(figsize=(10, 6))
sns.histplot(data=df_dataset_clothing, x='Cons_rating', kde=True)
plt.title('Rating distribution')
plt.show()
```
5. Generate WordCloud:
```python
word_could_dict = Counter(df_dataset_clothing['Cloth_class'].tolist())
wordcloud = WordCloud(width=1000, height=800).generate_from_frequencies(word_could_dict)
plt.figure(figsize=(16,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```

## 🌐 Deployment with Streamlit
To make this analysis interactive and accessible, we deploy the project using **Streamlit**.

### How to Deploy
1. Install Streamlit if not already installed:
   ```bash
   pip install streamlit
   ```
2. Create a Python script (e.g., `app.py`) and add the following code:
   ```python
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
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Open the provided URL in a browser to interact with the visualizations.

## 📊 Key Insights
- The dataset contains various clothing categories, each with different consumer ratings.
- Visualizations help understand rating distribution and consumer preferences.
- WordCloud provides an overview of frequently reviewed clothing categories.

## 📝 Future Improvements
- Perform sentiment analysis on customer reviews.
- Implement machine learning models for review classification.
- Expand dataset with additional clothing categories.

---
🔗 *Developed by Alvin Rahman Al Musyaffa*
