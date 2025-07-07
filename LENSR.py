# ========================== LENS eXpert Final Full Code ==========================

import streamlit as st
import pandas as pd
import requests
import joblib
import io
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# --- Config ---
st.set_page_config(page_title="LENSR eXpert", layout="wide")

# --- Header ---
st.markdown("""
    <div style='background: linear-gradient(90deg, #1e3c72, #2a5298); padding: 25px; border-radius: 15px; text-align: center;'>
        <h1 style='color: white; font-size: 48px;'>🔍 LENSR eXpert</h1>
        <h3 style='color: #e0f7fa;'>Smart NLP & Movie Recommendation Toolkit</h3>
        <p style='color: #ffffff; font-size: 18px;'>Created by <b style='color:#FFD700;'>Ritesh Kumar</b> | 📧 07mrriteshkr@gmail.com | 📍 Noida | 📞 6203727527</p>
    </div>
""", unsafe_allow_html=True)

# --- Load Models ---
spam_model = joblib.load("spam.pkl")
lan_model = joblib.load("lang_det.pkl")
review_model = joblib.load("review.pkl")
news_model = joblib.load("news_short.pkl")
movie_pkg = joblib.load("movie_recommendation.pkl")
movie_model = movie_pkg["model"]
vectorizer = movie_pkg["vectorizer"]
df5 = movie_pkg["df5"]

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
        <div style='margin-top: -15px; text-align: center;'>
            <img src='riteshsamridhipics.jpg' width='180'>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='
        background-color:#e0f0ff; 
        padding:8px 10px; 
        border-radius:10px; 
        margin-top:10px; 
        box-shadow:0 1px 5px rgba(0,0,0,0.1); 
        text-align:center;
        width:100%;
    '>
        <h4 style='color:#1e3c72; margin-bottom:4px; font-size:16px;'>🔬 LENSR NLP Toolkit</h4>
        <p style='color:#333333; font-size:12.5px; margin: 0;'>Explore different NLP & ML capabilities.</p>
    </div>
""", unsafe_allow_html=True)
    section = st.selectbox(
        "📌 Select a Project Module:",
        options=[
            "",  # Empty default
            "📨 Spam Classifier",
            "🌐 Language Detection",
            "🍽️ Review Sentiment",
            "🗞️ News Classifier",
            "🎬 Movie Recommendation",
            "🧠 Train Your Own Model"
        ],
        index=0,
        help="Choose an NLP or ML module to use"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
# --- Spam ---
if section == "📨 Spam Classifier":
    st.header("📨 Spam Classifier")
    st.markdown("""<p style='color:gray'>This module helps you detect whether a given message is spam or not using NLP techniques.</p>""", unsafe_allow_html=True)
    msg = st.text_input("📩 Enter a message:")
    if st.button("🔍 Predict One"):
        pred = spam_model.predict([msg])
        st.success("✅ Not Spam" if pred[0] else "❌ Spam")

    file = st.file_uploader("📁 Upload CSV/TXT file", type=["csv", "txt"])
    if file:
        df = pd.read_csv(file, header=None, names=["Msg"])
        df["Prediction"] = spam_model.predict(df["Msg"])
        df["Prediction"] = df["Prediction"].map({0: "❌ Spam", 1: "✅ Not Spam"})
        st.dataframe(df)

# --- Language ---
elif section == "🌐 Language Detection":
    st.header("🌐 Language Detection")
    st.markdown("""<p style='color:gray'>Automatically identify the language of your input text from various global languages.</p>""", unsafe_allow_html=True)
    msg = st.text_input("📝 Enter text:")
    if st.button("🔍 Detect One"):
        pred = lan_model.predict([msg])
        st.success(f"🌍 Language: {pred[0]}")

    file = st.file_uploader("📁 Upload file:", type=["csv", "txt"])
    if file:
        df = pd.read_csv(file, header=None, names=["Msg"])
        df["Prediction"] = lan_model.predict(df["Msg"])
        st.dataframe(df)

# --- Sentiment ---
elif section == "🍽️ Review Sentiment":
    st.header("🍽️ Food Review Sentiment")
    st.markdown("""<p style='color:gray'>Analyze customer reviews for positive or negative sentiment, perfect for restaurant feedback.</p>""", unsafe_allow_html=True)
    msg = st.text_input("🍔 Enter a review:")
    if st.button("🔍 Predict One"):
        pred = review_model.predict([msg])
        st.success("👍 Positive" if pred[0] else "👎 Negative")

    file = st.file_uploader("📁 Upload file:", type=["csv", "txt"])
    if file:
        df = pd.read_csv(file, header=None, names=["Msg"])
        df["Prediction"] = review_model.predict(df["Msg"])
        df["Prediction"] = df["Prediction"].map({0: "👎 Negative", 1: "👍 Positive"})
        st.dataframe(df)

# --- News ---
elif section == "🗞️ News Classifier":
    st.header("🗞️ News Classification")
    st.markdown("""<p style='color:gray'>Classify news headlines or content into categories such as politics, business, sports, etc.</p>""", unsafe_allow_html=True)
    msg = st.text_input("📰 Enter headline:")
    if st.button("🔍 Classify One"):
        pred = news_model.predict([msg])
        st.success(f"🗞️ Topic: {pred[0]}")

    file = st.file_uploader("📁 Upload file:", type=["csv", "txt"])
    if file:
        df = pd.read_csv(file, header=None, names=["Msg"])
        df["Prediction"] = news_model.predict(df["Msg"])
        st.dataframe(df)

# --- Movie Recommendation ---
elif section == "🎬 Movie Recommendation":
    st.header("🎬 Movie Recommendation System")
    st.markdown("""<p style='color:gray'>Select a movie and get top similar movie recommendations based on genre, cast, and other content-based filters.</p>""", unsafe_allow_html=True)
    df5['normalized_name'] = df5['name'].str.strip().str.lower()
    selected_movie = st.selectbox("🎬 Select a movie:", df5['name'].sort_values().unique(),index=None, placeholder="Choose a movie")
    if selected_movie:
        index = df5[df5['name'] == selected_movie].index[0]
        num_recommend = st.selectbox("🔢 Number of recommendations:", [2, 5, 7, 10], index=3)
        vectors = vectorizer.transform(df5['tag'])
        distances, indexes = movie_model.kneighbors(vectors[index], n_neighbors=num_recommend + 1)

        st.subheader("📽️ Top Recommended Movies")
        col1, col2, col3 = st.columns([1.1, 1, 1.1])
        col_index = 0
        rec_titles, rec_ids = [], []

        for i in indexes[0][1:]:
            title = df5.loc[i]['name']
            movie_id = df5.loc[i]['movie_id']
            genre = df5.loc[i].get('genre', 'N/A')
            director = df5.loc[i].get('director', 'N/A')
            cast = df5.loc[i].get('cast', 'N/A')

            url = f"http://www.omdbapi.com/?i={movie_id}&apikey=1d2456cd"
            try:
                data = requests.get(url).json()
                poster = data.get("Poster", "")
                rating = data.get("imdbRating", "N/A")
                year = data.get("Year", "N/A")
                director = data.get("Director", director)
                cast = data.get("Actors", cast)
                genre = data.get("Genre", genre)
            except:
                poster, rating, year = "", "N/A", "N/A"

            column = [col1, col2, col3][col_index % 3]
            card = f"""
                <div style='background:#f9f9f9; padding:10px; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.1); text-align:center; margin-bottom:15px;'>
                    <img src='{poster}' width='180px' height='270px' style='border-radius:8px;' /><br><br>
                    <b style='font-size:16px;'>{title}</b><br>
                    ⭐ {rating} | 📅 {year}<br>
                    🎬 {director}<br>
                    🎭 {cast}<br>
                    📚 {genre}
                </div>
            """
            column.markdown(card, unsafe_allow_html=True)
            col_index += 1
            rec_titles.append(title)
            rec_ids.append(movie_id)

        if rec_titles:
            df_result = pd.DataFrame({"Movie": rec_titles, "IMDB ID": rec_ids})
            buf = io.StringIO()
            df_result.to_csv(buf, index=False)
            st.download_button("📥 Download Recommendations", buf.getvalue(), file_name="recommended_movies.csv")

# --- Train Your Own Model ---
elif section == "🧠 Train Your Own Model":
    st.header("🧠 Train a Custom ML Model")
    st.markdown("""<p style='color:gray'>Upload your own dataset and train a text classification model using logistic regression or random forest.</p>""", unsafe_allow_html=True)

    file = st.file_uploader("📁 Upload your CSV or TXT file:", type=["csv", "txt"])
    if file:
        try:
            raw_bytes = file.read()
            try:
                raw_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raw_text = raw_bytes.decode("ISO-8859-1")
            file.seek(0)

            try:
                dialect = csv.Sniffer().sniff(raw_text[:1024])
                sep = dialect.delimiter
            except:
                sep = '\t' if '\t' in raw_text else (',' if ',' in raw_text else None)

            df = pd.read_csv(file, sep=sep, engine="python", on_bad_lines='skip')
        except Exception as e:
            st.error(f"❌ Could not read the file: {e}")
            df = None

        if df is not None:
            st.subheader("📊 Preview:")
            st.dataframe(df.head())
            st.write("Columns:", df.columns.tolist())
            st.write("Shape:", df.shape)
            st.write("Missing:", df.isnull().sum())

            all_cols = df.columns.tolist()
            target = st.selectbox("🎯 Target column", all_cols)
            model_choice = st.selectbox("🧠 Choose algorithm", ["Logistic Regression", "Random Forest"])
            text_col = st.selectbox("💬 Text column to vectorize", [col for col in all_cols if df[col].dtype == "object"])

            if st.button("🚀 Train Model"):
                try:
                    X_raw = df[text_col].fillna("")
                    y_raw = df[target]
                    le = LabelEncoder()
                    y = le.fit_transform(y_raw)

                    tf = TfidfVectorizer()
                    clf = LogisticRegression() if model_choice == "Logistic Regression" else RandomForestClassifier()
                    pipe = Pipeline([("tfidf", tf), ("clf", clf)])
                    scores = cross_validate(pipe, X_raw, y, cv=5, return_train_score=True)

                    st.success("✅ Training Complete!")
                    st.write("📈 Train Score:", scores["train_score"].mean())
                    st.write("📉 Test Score:", scores["test_score"].mean())

                    buf = io.BytesIO()
                    joblib.dump({"model": pipe, "label_encoder": le}, buf)
                    buf.seek(0)

                    st.download_button("📦 Download Trained Model", buf, file_name="custom_model.pkl")

                except Exception as e:
                    st.error(f"❌ Error during training: {e}")


with st.sidebar.expander("🌐 About Us"):
    st.markdown("""
    <div style='font-size: 15px;'>
        We are final-year students passionate about <b>Data Science</b> & <b>Natural Language Processing</b>.<br><br>
        This app is part of our academic project to help users test multiple NLP models with ease.
    </div>
    """, unsafe_allow_html=True)

with st.sidebar.expander("📞 Contact us"):
    st.write("📱 7061931957")
    st.write("✉️ 07nk05@gmail.com")

with st.sidebar.expander("🤝 Help & Instructions"):
    st.markdown("""
    <ul style='font-size: 14px;'>
        <li>Type or upload text to test the model.</li>
        <li>Use supported file formats: <b>.csv</b> or <b>.txt</b>.</li>
        <li>After prediction, download the result using the button.</li>
    </ul>
    """, unsafe_allow_html=True)

st.markdown("""
    <hr style="margin-top: 40px; border: none; border-top: 2px solid #ccc;" />
    <div style="text-align: center; padding: 10px; font-size: 24px; color: #555;">
        🔍 <b>LENSR eXpert</b> | Built with ❤️ using <b>Streamlit</b>, <b>sklearn</b>, and <b>nltk</b><br>
        📚 For Learning & Academic Use | © 2025
    </div>
""", unsafe_allow_html=True)
