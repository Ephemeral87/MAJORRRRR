import streamlit as st
import pandas as pd
from urllib.parse import urlparse, parse_qs
from langdetect import detect, DetectorFactory

from utils.youtube_api import get_video_comments
from utils.inference import predict_sentiment
from utils.charts import sentiment_pie, sentiment_by_language

# Ensure language detection is consistent
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
    # Quick check for Tamil script characters
    if any('\u0b80' <= char <= '\u0bff' for char in text):
        return "Tamil"
    try:
        lang = detect(text)
        # Map common codes to full names
        lang_map = {'en': 'English', 'ta': 'Tamil', 'hi': 'Hindi', 'te': 'Telugu'}
        return lang_map.get(lang, lang.upper())
    except:
        return "Unknown"

def refine_sentiment_accuracy(row):
    """
    Custom logic to fix errors where AI marks Tanglish requests as 'Negative'.
    """
    text = str(row['comment']).lower()
    
    # Keywords that usually indicate a request/question (Neutral), not a complaint (Negative)
    neutral_keywords = ['podunga', 'pannunga', 'sollunga', 'venum', 'review please', 'suggest', 'comparison', 'waiting', 'how to', 'next video']
    
    # Keywords/Emojis that are definitely positive
    positive_indicators = ['super', 'nice', 'mass', 'fire', '🔥', '👍', '❤️', 'love', 'thala', 'excellent', 'rocking']

    current_sentiment = row['sentiment']

    # Fix 1: If marked Negative but contains request keywords -> Neutral
    if current_sentiment == "Negative" and any(word in text for word in neutral_keywords):
        return "Neutral"
    
    # Fix 2: If marked Negative/Neutral but has positive indicators -> Positive
    if current_sentiment in ["Negative", "Neutral"] and any(word in text for word in positive_indicators):
        return "Positive"

    # Fix 3: Hard-coded Negative (Common Tamil/English slang for bad)
    if any(word in text for word in ['waste', 'worst', 'mokkai', 'bad', 'disappointing']):
        return "Negative"

    return current_sentiment

st.title("📊 YouTube Multilingual Sentiment Dashboard")
st.caption("Advanced research dashboard with Tanglish and Emoji support.")

with st.sidebar:
    st.header("Input Settings")
    api_key = st.text_input("YouTube API Key", type="password")
    youtube_url = st.text_input("YouTube Video URL")
    limit = st.slider("Comments to analyze", 20, 500, 100)
    st.divider()
    st.info("This dashboard uses XLM-RoBERTa Multilingual AI for high accuracy in mixed languages.")

run = st.button("Analyze Comments", use_container_width=True, type="primary")

if run:
    if not api_key or not youtube_url:
        st.warning("Please enter both API key and YouTube URL.")
        st.stop()

    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.error("Invalid YouTube URL.")
        st.stop()

    with st.spinner("Step 1: Fetching comments from YouTube..."):
        comments = get_video_comments(api_key, video_id, limit=limit)

    if isinstance(comments, dict) and "error" in comments:
        st.error(comments["error"])
        st.stop()

    if not comments:
        st.warning("No comments found for this video.")
        st.stop()

    # Create Dataframe
    df = pd.DataFrame(comments)
    
    with st.spinner("Step 2: Detecting languages and running Multilingual AI..."):
        # Detect Language
        df["language"] = df["comment"].astype(str).apply(detect_language_safe)
        
        # Run AI Sentiment
        sentiments, scores = predict_sentiment(df["comment"].astype(str).tolist())
        df["sentiment"] = sentiments
        df["confidence"] = scores
        
        # Apply the Refinement Fix for Accuracy
        df["sentiment"] = df.apply(refine_sentiment_accuracy, axis=1)

    # --- METRICS SECTION ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", len(df))
    
    pos_pct = (df['sentiment'].eq('Positive').mean() * 100)
    neu_pct = (df['sentiment'].eq('Neutral').mean() * 100)
    neg_pct = (df['sentiment'].eq('Negative').mean() * 100)
    
    c2.metric("Positive", f"{pos_pct:.1f}%")
    c3.metric("Neutral", f"{neu_pct:.1f}%")
    c4.metric("Negative", f"{neg_pct:.1f}%")
    c5.metric("Avg Confidence", f"{df['confidence'].mean():.2f}")

    tab1, tab2, tab3 = st.tabs(["📊 Overview Charts", "🔍 Comment Analysis", "💾 Raw Data"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(sentiment_pie(df), use_container_width=True)
        with col2:
            st.plotly_chart(sentiment_by_language(df), use_container_width=True)

    with tab2:
        selected_sentiment = st.selectbox(
            "Filter table by sentiment",
            ["All", "Positive", "Neutral", "Negative"]
        )

        temp_df = df.copy()
        if selected_sentiment != "All":
            temp_df = temp_df[temp_df["sentiment"] == selected_sentiment]

        # Apply coloring to the sentiment column for better visibility
        def color_sentiment(val):
            if val == 'Positive': color = '#2ecc71'
            elif val == 'Negative': color = '#e74c3c'
            else: color = '#3498db'
            return f'background-color: {color}; color: white; font-weight: bold'

        st.dataframe(
    temp_df[["author", "comment", "language", "sentiment", "confidence", "published_at"]].style.map(
        color_sentiment, subset=['sentiment']
    ),
    use_container_width=True
)

    with tab3:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results as CSV",
            csv,
            "youtube_sentiment_analysis.csv",
            "text/csv",
            use_container_width=True
        )
