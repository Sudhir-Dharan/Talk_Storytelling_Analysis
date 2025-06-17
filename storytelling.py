import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import openai
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from dotenv import load_dotenv  # Correct import for load_dotenv
import os
from transformers import pipeline  # Import Hugging Face pipeline for NLP tasks
import torch  # Import torch for device configuration
import json  # Import JSON for caching results
import time  # Import time module for measuring execution time
from sentence_transformers import SentenceTransformer, util  # Ensure this import works after installation

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')  # Explicitly specify the .env file path
load_dotenv(dotenv_path)
openai_api_key = os.getenv("STORY_OPENAI_API_KEY")

if not openai_api_key:
    print("Error: OpenAI API key not found. Ensure the .env file is correctly configured.")
else:
    print("OpenAI API Key loaded successfully.")  # Debugging line to confirm the key is loaded

# GPT 4.0+ compatible client initialization
client = openai.OpenAI(api_key=openai_api_key)

# Initialize sentiment analysis pipeline with an explicit model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU
)

# Initialize zero-shot classification pipeline with an explicit model
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise CPU
)

# Initialize SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and fast model

st.set_page_config(page_title=" Talk Storytelling Analyzer", layout="wide")
st.title("🎙️ Talk Storytelling Analyzer")

# Directory to store cached results
cache_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(cache_dir, exist_ok=True)  # Ensure the directory exists

def extract_video_id_and_title(line):
    match = re.search(r"v=([\w-]{11})", line)
    if match:
        video_id = match.group(1)
        return video_id, f"Talk {video_id}"
    elif "|" in line:
        return [x.strip() for x in line.split("|", 1)]
    else:
        raise ValueError("Unable to parse line. Please ensure it contains a video ID or YouTube URL.")

def get_transcript(video_id):
    try:
        # Check if the transcript is already cached
        cached_result = load_cached_result(video_id)
        if cached_result and "Transcript" in cached_result:
            return cached_result["Transcript"]

        # Fetch the transcript from YouTube
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def cache_result(video_id, result):
    """Cache the analysis result for a video."""
    cache_file = os.path.join(cache_dir, f"{video_id}.json")
    with open(cache_file, "w") as f:
        json.dump(result, f)

def load_cached_result(video_id):
    """Load the cached analysis result for a video if it exists."""
    cache_file = os.path.join(cache_dir, f"{video_id}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return None

def classify_with_embeddings(sentences, candidate_labels, threshold=0.3):  # Lowered threshold to 0.3
    """Classify sentences using embeddings and cosine similarity."""
    if not sentences or not candidate_labels:
        return {}  # Return empty if inputs are invalid

    # Generate embeddings for sentences and candidate labels
    sentence_embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    label_embeddings = embedding_model.encode(candidate_labels, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_scores = util.cos_sim(sentence_embeddings, label_embeddings)

    # Debugging: Print cosine similarity scores
    print("Cosine Similarity Scores:", cosine_scores)

    # Aggregate results
    detected_labels = {}
    for i, sentence_scores in enumerate(cosine_scores):
        for j, score in enumerate(sentence_scores):
            if score > threshold:  # Threshold for considering a label
                label = candidate_labels[j]
                detected_labels[label] = detected_labels.get(label, 0) + score.item()

    return detected_labels

def analyze_transcript(transcript, video_id):
    try:
        # Check if the result is already cached
        cached_result = load_cached_result(video_id)
        if cached_result:
            return cached_result

        # Tokenize transcript into sentences for analysis
        sentences = re.split(r'(?<=[.!?]) +', transcript)

        # Debugging: Print sentences to ensure they are being tokenized correctly
        print("Tokenized Sentences:", sentences)

        # Define candidate labels for emotions, conflicts, and storytelling tags
        emotion_labels = [
            "love", "fear", "hope", "failure", "success", "vulnerability", 
            "joy", "sadness", "anger", "trust", "anticipation", "surprise", 
            "disgust", "pride", "shame", "gratitude", "envy", "compassion"
        ]
        conflict_labels = [
            "problem", "struggle", "challenge", "conflict", "obstacle", 
            "disagreement", "tension", "fight", "rivalry", "misunderstanding"
        ]
        storytelling_labels = [
            "personal story", "humor", "data-driven", "inspirational", 
            "educational", "motivational", "narrative", "persuasive", 
            "entertaining", "informative", "emotional", "thought-provoking"
        ]

        # Perform classification using embeddings and cosine similarity
        detected_emotions = classify_with_embeddings(sentences, emotion_labels, threshold=0.3)
        detected_conflicts = classify_with_embeddings(sentences, conflict_labels, threshold=0.3)
        detected_tags = classify_with_embeddings(sentences, storytelling_labels, threshold=0.3)

        # Debugging: Print detected labels
        print("Detected Emotions:", detected_emotions)
        print("Detected Conflicts:", detected_conflicts)
        print("Detected Tags:", detected_tags)

        # Convert cumulative scores to sorted lists
        sorted_emotions = sorted(detected_emotions.items(), key=lambda x: x[1], reverse=True)
        sorted_conflicts = sorted(detected_conflicts.items(), key=lambda x: x[1], reverse=True)
        sorted_tags = sorted(detected_tags.items(), key=lambda x: x[1], reverse=True)

        # Prepare results as strings with scores
        emotion_keywords = ", ".join([f"{label} ({score:.2f})" for label, score in sorted_emotions])
        conflict_keywords = ", ".join([f"{label} ({score:.2f})" for label, score in sorted_conflicts])
        storytelling_tags = ", ".join([f"{label} ({score:.2f})" for label, score in sorted_tags])

        # Aggregate sentiment analysis
        sentiment_results = sentiment_analyzer(sentences)
        positive_count = sum(1 for result in sentiment_results if result['label'] == 'POSITIVE')
        negative_count = sum(1 for result in sentiment_results if result['label'] == 'NEGATIVE')
        neutral_count = len(sentiment_results) - positive_count - negative_count

        # Prepare the result
        result = {
            "Transcript": transcript,
            "Length (words)": len(transcript.split()),
            "Positive Sentences": positive_count,
            "Negative Sentences": negative_count,
            "Neutral Sentences": neutral_count,
            "Emotion Keywords": emotion_keywords,
            "Conflict Words": conflict_keywords,
            "Storytelling Tags": storytelling_tags
        }

        # Cache the result
        cache_result(video_id, result)
        return result
    except Exception as e:
        st.error(f"Error analyzing transcript: {e}")
        return None

def gpt_analysis(transcript, emotion_keywords, conflict_keywords, storytelling_tags):
    try:
        prompt = f"""
        Analyze this Talk transcript for storytelling elements like protagonist, antagonist, challenge, action, and success.
        Additionally, consider the following extracted keywords and their scores:

        - Emotion Keywords: {emotion_keywords}
        - Conflict Words: {conflict_keywords}
        - Storytelling Tags: {storytelling_tags}

        Transcript:
        {transcript[:2000]}  # Limiting to 2000 characters for GPT input
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a storytelling expert."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT Error: {e}"

def plot_tag_frequencies(df):
    """Plot separate weighted frequencies for emotions, conflicts, and storytelling tags."""
    # Initialize dictionaries for emotions, conflicts, and storytelling tags
    emotion_scores = {}
    conflict_scores = {}
    storytelling_scores = {}

    # Process each row to extract and aggregate scores
    for _, row in df.iterrows():
        if row["Emotion Keywords"]:
            for tag in row["Emotion Keywords"].split(", "):
                label, score = tag.rsplit(" (", 1)
                score = float(score.strip(")"))
                emotion_scores[label] = emotion_scores.get(label, 0) + score
        if row["Conflict Words"]:
            for tag in row["Conflict Words"].split(", "):
                label, score = tag.rsplit(" (", 1)
                score = float(score.strip(")"))
                conflict_scores[label] = conflict_scores.get(label, 0) + score
        if row["Storytelling Tags"]:
            for tag in row["Storytelling Tags"].split(", "):
                label, score = tag.rsplit(" (", 1)
                score = float(score.strip(")"))
                storytelling_scores[label] = storytelling_scores.get(label, 0) + score

    # Convert to sorted series for plotting
    emotion_series = pd.Series(emotion_scores).sort_values(ascending=False)
    conflict_series = pd.Series(conflict_scores).sort_values(ascending=False)
    storytelling_series = pd.Series(storytelling_scores).sort_values(ascending=False)

    # Plot emotions
    if not emotion_series.empty:
        st.subheader("📊 Weighted Emotions")
        fig, ax = plt.subplots()
        emotion_series.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_ylabel("Weighted Score")
        ax.set_xlabel("Emotions")
        ax.set_title("Weighted Frequencies of Emotions")
        st.pyplot(fig)

    # Plot conflicts
    if not conflict_series.empty:
        st.subheader("📊 Weighted Conflicts")
        fig, ax = plt.subplots()
        conflict_series.plot(kind='bar', ax=ax, color='salmon')
        ax.set_ylabel("Weighted Score")
        ax.set_xlabel("Conflicts")
        ax.set_title("Weighted Frequencies of Conflicts")
        st.pyplot(fig)

    # Plot storytelling tags
    if not storytelling_series.empty:
        st.subheader("📊 Weighted Storytelling Tags")
        fig, ax = plt.subplots()
        storytelling_series.plot(kind='bar', ax=ax, color='lightgreen')
        ax.set_ylabel("Weighted Score")
        ax.set_xlabel("Storytelling Tags")
        ax.set_title("Weighted Frequencies of Storytelling Tags")
        st.pyplot(fig)

    # Return all scores for further use
    return {
        "emotions": emotion_scores,
        "conflicts": conflict_scores,
        "storytelling_tags": storytelling_scores
    }

def generate_comprehensive_report(df, tag_scores):
    """Generate a comprehensive report using weighted tag scores."""
    try:
        st.subheader("📋 Comprehensive Report")
        good_points = []
        improvement_areas = []
        data_learnings = []

        # Summarize learnings based on weighted scores
        top_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        top_tags_summary = ", ".join([f"{tag} ({score:.2f})" for tag, score in top_tags])

        for _, row in df.iterrows():
            if "humor" in row["Storytelling Tags"]:
                good_points.append(f"{row['Title']}: Effective use of humor.")
            if "data-driven" in row["Storytelling Tags"]:
                good_points.append(f"{row['Title']}: Strong use of data to support storytelling.")
            if not row["Emotion Keywords"]:
                improvement_areas.append(f"{row['Title']}: Could include more emotional elements.")
            if not row["Conflict Words"]:
                improvement_areas.append(f"{row['Title']}: Could include more conflict or challenges to engage the audience.")

            data_learnings.append(f"{row['Title']}: {row['Storytelling Tags']}")

        st.markdown("### ✅ Good Points")
        st.markdown("\n".join(f"- {point}" for point in good_points) or "No significant good points identified.")

        st.markdown("### ⚠️ Improvement Areas")
        st.markdown("\n".join(f"- {area}" for area in improvement_areas) or "No significant improvement areas identified.")

        st.markdown("### 📊 Data-Driven Learnings")
        st.markdown("\n".join(f"- {learning}" for learning in data_learnings) or "No significant data-driven learnings identified.")

        st.markdown("### 🔝 Top Weighted Tags")
        st.markdown(top_tags_summary or "No significant tags identified.")
    except Exception as e:
        st.error(f"Error generating comprehensive report: {e}")

def generate_comprehensive_report_with_gpt(df, tag_scores):
    """Generate a GPT-based comprehensive report using tag scores and consolidated learnings."""
    try:
        st.subheader("📋 Comprehensive Report (GPT-Generated)")

        # Combine transcripts for GPT input
        combined_transcripts = " ".join(df["Transcript"].dropna().values)

        # Flatten tag_scores and prepare top tags summary
        flattened_scores = {**tag_scores["emotions"], **tag_scores["conflicts"], **tag_scores["storytelling_tags"]}
        top_tags = sorted(flattened_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        top_tags_summary = ", ".join([f"{tag} ({score:.2f})" for tag, score in top_tags])

        # GPT prompt with consolidated learnings and tag scores
        prompt = f"""
        Based on the following combined transcripts of analyzed talks and the extracted weighted tags, provide a comprehensive report:
        - Highlight the good storytelling practices observed.
        - Insights based on the weighted storytelling tags, emotions, and conflicts.
        - Derive generic data-driven learnings applicable across all talks.

        Top Weighted Tags:
        {top_tags_summary}

        Combined Transcripts:
        {combined_transcripts[:3000]}  # Limiting to 3000 characters for GPT input
        """
        with st.spinner("Generating comprehensive report using GPT..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are expert at analyzing data & providing insights."},
                    {"role": "user", "content": prompt}
                ]
            )
            report = response.choices[0].message.content

        st.markdown(report)
    except Exception as e:
        st.error(f"Error generating GPT-based comprehensive report: {e}")

def generate_generic_gpt_prompt(df):
    try:
        st.subheader("🛠️ Generic GPT Prompt for Storytelling")
        combined_tags = ", ".join(df["Storytelling Tags"].dropna().values)
        prompt = f"""
        Create a custom GPT model prompt for storytelling analysis based on the following observed storytelling elements:
        {combined_tags}

        The model should:
        - Identify storytelling elements such as protagonist, antagonist, challenges, actions, and resolutions.
        - Provide suggestions for improving emotional engagement and conflict resolution.
        - Highlight data-driven insights to enhance storytelling effectiveness.
        """
        st.text_area("Custom GPT Prompt", prompt, height=200)
    except Exception as e:
        st.error(f"Error generating generic GPT prompt: {e}")

def render_wordcloud(df):
    """Generate and display a word cloud from the transcripts."""
    all_text = " ".join(df["Transcript"].dropna().values)
    if all_text:
        st.subheader("☁️ Word Cloud of Full Transcripts")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

with st.form("input_form"):
    st.subheader("Enter  Talks")
    talk_entries = st.text_area("Enter  Talk YouTube IDs and Titles (one per line, format: video_id | title)", height=200)
    use_gpt = st.checkbox("Include GPT storytelling insights (needs API key)", value=False)
    submit = st.form_submit_button("Analyze Talks")

if submit:
    if not talk_entries.strip():
        st.warning("Please enter at least one Talk.")
    else:
        overall_start_time = time.time()  # Start timer for the entire computation
        data = []
        for line in talk_entries.strip().splitlines():
            try:
                video_start_time = time.time()  # Start timer for each video
                video_id, title = extract_video_id_and_title(line)
                youtube_url = f"https://www.youtube.com/watch?v={video_id}"
                st.markdown(f"### 📽️ [{title}]({youtube_url})")
                transcript = get_transcript(video_id)
                if transcript:
                    basic = analyze_transcript(transcript, video_id)
                    for k, v in basic.items():
                        if k != "Transcript":
                            st.write(f"**{k}:** {v}")
                    if use_gpt:
                        with st.spinner("Running GPT analysis..."):
                            insights = gpt_analysis(
                                transcript,
                                basic["Emotion Keywords"],
                                basic["Conflict Words"],
                                basic["Storytelling Tags"]
                            )
                            st.markdown("**🔍 Insights:**")
                            st.markdown(insights)
                    data.append({
                        "Video ID": video_id,
                        "Title": title,
                        **basic
                    })
                video_end_time = time.time()  # End timer for each video
                st.write(f"⏱️ Time taken for video '{title}': {video_end_time - video_start_time:.2f} seconds")
            except Exception as e:
                st.error(f"Could not process line: {line} — {e}")

        if data:
            df = pd.DataFrame(data)
            tag_scores = plot_tag_frequencies(df)  # Get weighted scores
            st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="storytelling_analysis.csv")
            render_wordcloud(df)
            generate_comprehensive_report_with_gpt(df, tag_scores)  # Pass weighted scores to the report
            #generate_generic_gpt_prompt(df)

        overall_end_time = time.time()  # End timer for the entire computation
        st.write(f"⏱️ Total time taken for all computations: {overall_end_time - overall_start_time:.2f} seconds")
