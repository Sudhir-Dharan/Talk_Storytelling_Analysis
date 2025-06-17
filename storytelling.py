import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import openai
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# GPT 4.0+ compatible client initialization
client = openai.OpenAI(api_key="<OPENAI-KEY-HERE>")



st.set_page_config(page_title=" Talk Storytelling Analyzer", layout="wide")
st.title("🎙️ Talk Storytelling Analyzer")

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
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def analyze_transcript(transcript):
    words = transcript.split()
    word_count = len(words)
    emotions = [w for w in ["love", "fear", "hope", "failure", "success", "vulnerable"] if w in transcript.lower()]
    conflicts = [w for w in ["problem", "struggle", "challenge", "conflict"] if w in transcript.lower()]
    tags = []
    if "story" in transcript.lower():
        tags.append("personal story")
    if "funny" in transcript.lower() or "laugh" in transcript.lower():
        tags.append("humor")
    if "data" in transcript.lower():
        tags.append("data-driven")

    return {
        "Transcript": transcript,
        "Length (words)": word_count,
        "Emotion Keywords": ", ".join(emotions),
        "Conflict Words": ", ".join(conflicts),
        "Storytelling Tags": ", ".join(tags)
    }

def gpt_analysis(transcript):
    try:
        prompt = f"""
        Analyze this  Talk transcript for storytelling elements like protagonist, antogonist, challege, action, and success.:

        {transcript[:2000]}
        """
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a  storytelling expert."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT Error: {e}"

def plot_tag_frequencies(df):
    all_tags = ",".join(df["Storytelling Tags"].dropna()).split(",")
    tag_series = pd.Series([t.strip() for t in all_tags if t.strip()])
    freq = tag_series.value_counts()
    if not freq.empty:
        st.subheader("📊 Storytelling Tags Frequency")
        fig, ax = plt.subplots()
        freq.plot(kind='bar', ax=ax)
        st.pyplot(fig)

def render_wordcloud(df):
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
        st.warning("Please enter at least one  Talk.")
    else:
        data = []
        for line in talk_entries.strip().splitlines():
            try:
                video_id, title = extract_video_id_and_title(line)
                youtube_url = f"https://www.youtube.com/watch?v={video_id}"
                st.markdown(f"### 📽️ [{title}]({youtube_url})")
                transcript = get_transcript(video_id)
                if transcript:
                    basic = analyze_transcript(transcript)
                    for k, v in basic.items():
                        if k != "Transcript":
                            st.write(f"**{k}:** {v}")
                    if use_gpt:
                        with st.spinner("Running GPT analysis..."):
                            insights = gpt_analysis(transcript)
                            st.markdown("**🔍 Insights:**")
                            st.markdown(insights)
                    data.append({
                        "Video ID": video_id,
                        "Title": title,
                        **basic
                    })
            except Exception as e:
                st.error(f"Could not process line: {line} — {e}")

        if data:
            df = pd.DataFrame(data)
            st.download_button("📥 Download CSV", df.to_csv(index=False), file_name="storytelling_analysis.csv")
            plot_tag_frequencies(df)
            render_wordcloud(df)
