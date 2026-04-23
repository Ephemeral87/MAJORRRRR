import streamlit as st
import pandas as pd
from urllib.parse import urlparse, parse_qs
from langdetect import detect, DetectorFactory

from utils.youtube_api import get_video_comments
from utils.inference import predict_sentiment
from utils.charts import sentiment_pie, sentiment_by_language

# Set seed for consistent language detection
DetectorFactory.seed = 0

st.set_page_config(
    page_title="Multilingual Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- HELPER FUNCTIONS ---

def extract_video_id(url):
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path.strip("/")
    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [None])[0]
    return None

def detect_language_safe(text):
    """
    Detects language using Unicode script ranges for Indian languages
    and langdetect for Romanized (English-script) languages.
    """
    # Native Script Detection
    if any('\u0b80' <= char <= '\u0bff' for char in text): return "Tamil"
    if any('\u0900' <= char <= '\u097f' for char in text): return "Hindi"
    if any('\u0c00' <= char <= '\u0c7f' for char in text): return "Telugu"
    if any('\u0c80' <= char <= '\u0cff' for char in text): return "Kannada"
    
    try:
        lang = detect(text)
        lang_map = {
            'en': 'English', 'ta': 'Tamil (English Script)', 
            'hi': 'Hindi (English Script)', 'te': 'Telugu', 'kn': 'Kannada'
        }
        return lang_map.get(lang, lang.upper())
    except:
        return "Unknown"

def refine_sentiment_accuracy(row):
    """
    Manual override logic for higher accuracy in South Asian contexts.
    Moves 'Requests' from Negative to Neutral.
    """
    text = str(row['comment']).lower()
    
    # 1. Neutral Request Keywords (AI often misinterprets these as Negative)
    neutral_keywords = [
        # Tamil
        'podunga', 'pannunga', 'sollunga', 'venum', 'kodunga',
        # Hindi
        'dikhao', 'batiye', 'chahiye', 'karo', 'banao', 'do',
        # Telugu
        'cheyyandi', 'chupinchandi', 'eppudu', 'cheppandi',
        # Kannada
        'madi', 'maadi', 'torisi', 'heli', 'beku',
        # English
        'please', 'suggest', 'review', 'comparison', 'waiting'
    ]
    
    # 2. Positive Slang & Emojis
    positive_indicators = [
        'super', 'nice', 'mass', 'fire', '🔥', '👍', '❤️', 'love', 'excellent',
        'achha', 'badhiya', 'mast', 'bagundi', 'baagundi', 'chennagide'
    ]

    current_sentiment = row['sentiment']

    # Logic: If AI says Negative but user is just asking for a video -> Neutral
    if current_sentiment == "Negative" and any(word in text for word in neutral_keywords):
        return "Neutral"
    
    # Logic: If AI is unsure but there are positive emojis/words -> Positive
    if current_sentiment in ["Negative", "Neutral"] and any(word in text for word in positive_indicators):
        return "Positive"

    # Logic: Hard-coded Negatives
    if any(word in text for word in ['waste', 'worst', 'mokkai', 'bekar', 'chetha', 'kharaab', 'bad']):
        return "Negative"

    return current_sentiment

# --- UI LAYOUT ---

st.title("📊 YouTube Multilingual Sentiment Dashboard")
st.caption("Research Tool for English, Tamil, Hindi, Telugu, and Kannada Sentiment Analysis.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("YouTube API Key", type="password", help="Get this from Google Cloud Console")
    youtube_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    limit = st.slider("Comments to analyze", 20, 500, 100)
    st.divider()
    st.info("Supported: Native scripts (தமிழ், हिंदी, తెలుగు, ಕನ್ನಡ) and Romanized (Tanglish, Hinglish, etc.)")

run = st.button("Run Multilingual Analysis", use_container_width=True, type="primary")

if run:
    if not api_key or not youtube_url:
        st.warning("Please provide both an API Key and a Video URL.")
        st.stop()

    video_id = extract_video_id(youtube_url)
    if not video_id:
        st.error("Invalid YouTube URL format.")
        st.stop()

    # 1. Fetch Comments
    with st.spinner("Fetching comments from YouTube..."):
        comments = get_video_comments(api_key, video_id, limit=limit)

    if isinstance(comments, dict) and "error" in comments:
        st.error(f"API Error: {comments['error']}")
        st.stop()

    if not comments:
        st.warning("No comments found.")
        st.stop()

    # 2. Process Data
    df = pd.DataFrame(comments)
    
    with st.spinner("Running AI Sentiment Analysis..."):
        # Detect Language
        df["language"] = df["comment"].astype(str).apply(detect_language_safe)
        
        # Predict Sentiment (using XLM-RoBERTa Multilingual)
        sentiments, scores = predict_sentiment(df["comment"].astype(str).tolist())
        df["sentiment"] = sentiments
        df["confidence"] = scores
        
        # Apply Custom Refinement for high accuracy
        df["sentiment"] = df.apply(refine_sentiment_accuracy, axis=1)

    # 3. Display Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Comments", len(df))
    m2.metric("Positive", f"{(df['sentiment'].eq('Positive').mean()*100):.1f}%")
    m3.metric("Neutral", f"{(df['sentiment'].eq('Neutral').mean()*100):.1f}%")
    m4.metric("Negative", f"{(df['sentiment'].eq('Negative').mean()*100):.1f}%")

    # 4. Tabs for Visuals and Data
    tab1, tab2, tab3 = st.tabs(["📈 Visualization", "🔍 Detailed Analysis", "💾 Export Data"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(sentiment_pie(df), use_container_width=True)
        with c2:
            st.plotly_chart(sentiment_by_language(df), use_container_width=True)

    with tab2:
        # Sentiment Color Styling
        def color_sentiment(val):
            if val == 'Positive': color = '#2ecc71'
            elif val == 'Negative': color = '#e74c3c'
            else: color = '#3498db'
            return f'background-color: {color}; color: white; font-weight: bold'

        # Filter option
        filter_choice = st.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"])
        display_df = df.copy()
        if filter_choice != "All":
            display_df = display_df[display_df["sentiment"] == filter_choice]

        st.dataframe(
            display_df[["author", "comment", "language", "sentiment", "confidence"]].style.map(
                color_sentiment, subset=['sentiment']
            ), 
            use_container_width=True
        )

    with tab3:
        st.subheader("Download Research Results")
        st.write("The CSV file includes original comments, detected languages, sentiment labels, and AI confidence scores.")
        
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV File",
            data=csv,
            file_name=f"youtube_analysis_{video_id}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.write("### Data Preview")
        st.dataframe(df.head(10))
