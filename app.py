import streamlit as st
import pandas as pd
from urllib.parse import urlparse, parse_qs
from langdetect import detect, DetectorFactory

from utils.youtube_api import get_video_comments
from utils.inference import predict_sentiment
from utils.charts import sentiment_pie, sentiment_by_language

DetectorFactory.seed = 0

st.set_page_config(
    page_title="YouTube Multilingual Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)


def extract_video_id(url):
    parsed = urlparse(url)

    if "youtu.be" in parsed.netloc:
        return parsed.path.strip("/")

    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [None])[0]

    return None


def detect_language_safe(text):
    try:
        return detect(text)
    except:
        return "unknown"


st.title("📊 YouTube Multilingual Sentiment Dashboard")
st.caption("Research dashboard for multilingual YouTube comment sentiment analysis.")

with st.sidebar:
    st.header("Input")
    api_key = st.text_input("YouTube API Key", type="password")
    youtube_url = st.text_input("YouTube Video URL")
    limit = st.slider("Comments to analyze", 20, 300, 100)

run = st.button("Analyze Comments", use_container_width=True)

if run:
    if not api_key or not youtube_url:
        st.warning("Enter both API key and YouTube URL.")
        st.stop()

    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.error("Invalid YouTube URL.")
        st.stop()

    with st.spinner("Fetching YouTube comments..."):
        comments = get_video_comments(api_key, video_id, limit=limit)

    if isinstance(comments, dict) and "error" in comments:
        st.error(comments["error"])
        st.stop()

    if not comments:
        st.warning("No comments found.")
        st.stop()

    df = pd.DataFrame(comments)
    df["language"] = df["comment"].astype(str).apply(detect_language_safe)

    with st.spinner("Running sentiment model..."):
        sentiments, scores = predict_sentiment(df["comment"].astype(str).tolist())
        df["sentiment"] = sentiments
        df["confidence"] = scores

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Comments", len(df))
    c2.metric("Positive %", f"{(df['sentiment'].eq('Positive').mean() * 100):.1f}")
    c3.metric("Negative %", f"{(df['sentiment'].eq('Negative').mean() * 100):.1f}")
    c4.metric("Avg Confidence", f"{df['confidence'].mean():.2f}")

    tab1, tab2, tab3 = st.tabs(["Overview", "Comment Analysis", "Raw Data"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(sentiment_pie(df), use_container_width=True)
        with col2:
            st.plotly_chart(sentiment_by_language(df), use_container_width=True)

    with tab2:
        selected_sentiment = st.selectbox(
            "Filter by sentiment",
            ["All", "Positive", "Neutral", "Negative", "Other"]
        )

        temp_df = df.copy()
        if selected_sentiment != "All":
            temp_df = temp_df[temp_df["sentiment"] == selected_sentiment]

        st.dataframe(
            temp_df[["author", "comment", "language", "sentiment", "confidence", "published_at"]],
            use_container_width=True
        )

    with tab3:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv,
            "youtube_sentiment_results.csv",
            "text/csv"
        )
