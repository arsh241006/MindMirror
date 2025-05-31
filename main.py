import sqlite3
import datetime
import random
import streamlit as st
import matplotlib.pyplot as plt
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
import re


# === Setup database ===
conn = sqlite3.connect("journal.db", check_same_thread=False)
cursor = conn.cursor()

# === Ensure DB Schema ===


def ensure_correct_schema():
    cursor.execute("PRAGMA table_info(journal)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    if 'id' not in column_names:
        cursor.execute("ALTER TABLE journal RENAME TO journal_old")
        cursor.execute('''
            CREATE TABLE journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                entry TEXT,
                sentiment TEXT,
                emotion TEXT
            )
        ''')
        cursor.execute('''
            INSERT INTO journal (date, entry, sentiment, emotion)
            SELECT date, entry, sentiment, emotion FROM journal_old
        ''')
        cursor.execute("DROP TABLE journal_old")
        conn.commit()


cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name='journal'")
if cursor.fetchone():
    ensure_correct_schema()
else:
    cursor.execute('''
        CREATE TABLE journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            entry TEXT,
            sentiment TEXT,
            emotion TEXT
        )
    ''')
    conn.commit()

# === Emotion Model Setup ===
emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

# === Helper Functions ===


def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"


def get_dominant_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return labels[torch.argmax(probs).item()].capitalize()


def get_random_quote():
    return random.choice([
        "Keep going, you're doing great!",
        "Every day is a second chance.",
        "You're capable of amazing things!",
        "Be proud of how far you've come.",
        "One step at a time."
    ])


def extract_keywords(entries, top_n=20):
    words = []
    for entry in entries:
        words.extend(re.findall(r'\b\w{4,}\b', entry.lower()))
    return Counter(words).most_common(top_n)


# === Streamlit App ===
st.set_page_config(page_title="MindMirror", layout="centered")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode


# Apply Theme
is_dark = st.session_state.dark_mode
text_color = 'white' if is_dark else '#111111'
bg_color = 'rgba(18,18,18,0.85)' if is_dark else 'rgba(255,255,255,0.85)'
highlight_color = '#ffcc70' if is_dark else '#333333'

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('https://images.unsplash.com/photo-1523240795612-9a054b0db644');
        background-size: cover;
        background-attachment: fixed;
        color: {text_color};
    }}
    .block-container {{
        backdrop-filter: blur(6px);
        background-color: {bg_color};
        border-radius: 12px;
        padding: 2rem;
        color: {text_color};
    }}
    input, textarea, .stTextArea textarea {{
        color: {text_color} !important;
        background-color: {'#333' if is_dark else 'white'} !important;
    }}
    .stAlert, .stInfo, .stMarkdown p, .stMarkdown div {{
        color: {text_color} !important;
    }}
    h1, h2, h3, h4 {{
        color: {highlight_color};
    }}
    .stButton>button {{
        background-color: {'#4CAF50' if is_dark else '#007ACC'};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 16px;
    }}
    .stButton>button:hover {{
        background-color: {'#45a049' if is_dark else '#005fa3'};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# UI
st.title("🧠 MindMirror - Mood Journal")
st.button("🌙 Toggle Dark Mode" if not is_dark else "☀ Toggle Light Mode",
          on_click=toggle_theme)

st.markdown("### 💡 Daily Motivation")
st.markdown(
    f"<div style='color:{text_color}'>{get_random_quote()}</div>", unsafe_allow_html=True)

menu = st.sidebar.radio("Choose an option", [
    "Write new journal entry", "View all entries", "Show mood trend", "Weekly/Monthly Summary"
])

if menu == "Write new journal entry":
    st.subheader("📔 New Journal Entry")
    entry = st.text_area("Write about your day:")
    if st.button("Analyze and Save"):
        if entry.strip():
            sentiment = get_sentiment(entry)
            try:
                emotion = get_dominant_emotion(entry)
            except:
                emotion = "Unknown"
            date = datetime.date.today().isoformat()
            cursor.execute("INSERT INTO journal (date, entry, sentiment, emotion) VALUES (?, ?, ?, ?)",
                           (date, entry, sentiment, emotion))
            conn.commit()
            st.success(f"Saved! Sentiment: {sentiment}, Emotion: {emotion}")
        else:
            st.warning("Please enter something.")

elif menu == "View all entries":
    st.subheader("📚 All Journal Entries")
    cursor.execute(
        "SELECT id, date, entry, sentiment, emotion FROM journal ORDER BY date DESC")
    rows = cursor.fetchall()
    if not rows:
        st.info("No entries yet.")
    else:
        for entry_id, date, entry, sentiment, emotion in rows:
            with st.expander(f"📅 {date} | {sentiment} | {emotion}"):
                st.markdown(
                    f"<div style='color:{text_color}'>{entry}</div>", unsafe_allow_html=True)

elif menu == "Show mood trend":
    st.subheader("📈 Mood Trend Over Time")
    cursor.execute("SELECT date, sentiment FROM journal")
    data = cursor.fetchall()
    if not data:
        st.info("No data to plot.")
    else:
        df = pd.DataFrame(data, columns=["Date", "Sentiment"])
        df["Mood"] = df["Sentiment"].map(
            {"Positive": 1, "Neutral": 0, "Negative": -1})
        df["Date"] = pd.to_datetime(df["Date"])
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, ax = plt.subplots()
        ax.plot(df["Date"], df["Mood"], marker='o',
                color='#ffcc70' if is_dark else '#007ACC')
        ax.set_xlabel("Date")
        ax.set_ylabel("Mood")
        ax.set_title("Mood Trend")
        ax.grid(True)
        st.pyplot(fig)

elif menu == "Weekly/Monthly Summary":
    st.subheader("📊 Summary")
    range_option = st.selectbox(
        "Select Time Range", ["Last 7 Days", "Last 30 Days"])
    delta_days = 7 if range_option == "Last 7 Days" else 30
    start_date = datetime.date.today() - datetime.timedelta(days=delta_days)
    cursor.execute(
        "SELECT date, entry, sentiment, emotion FROM journal WHERE date >= ?", (start_date.isoformat(),))
    rows = cursor.fetchall()
    if not rows:
        st.info("No entries found in the selected time range.")
    else:
        df = pd.DataFrame(
            rows, columns=["Date", "Entry", "Sentiment", "Emotion"])
        df["Date"] = pd.to_datetime(df["Date"])
        st.markdown(
            f"<div style='color:{text_color}'>📝 Total entries: {len(df)}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:{text_color}'>❤ Most common sentiment: {df['Sentiment'].mode()[0]}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:{text_color}'>😊 Most common emotion: {df['Emotion'].mode()[0]}</div>", unsafe_allow_html=True)
        st.bar_chart(df["Emotion"].value_counts())
        wordcloud = WordCloud(width=800, height=400, background_color='black' if is_dark else 'white').generate(
            " ".join(df["Entry"]))
        st.image(wordcloud.to_array(),
                 caption="😇 Most Frequent Words", use_container_width=True)
