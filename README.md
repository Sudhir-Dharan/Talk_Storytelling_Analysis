# 🗂 Talk Storytelling Analyzer

This document outlines the structure, purpose, and technical components of the `Talk_Storytelling_Analysis` project, which is a lightweight app designed to mine storytelling patterns from TED Talks using YouTube transcripts.

---

## 📌 Project Purpose

The project aims to:

* Help users understand what makes a talk compelling from a narrative perspective.
* Use natural language processing and optional GPT analysis to break down story structure.
* Provide an intuitive Streamlit UI to analyze TED Talks through storytelling lenses like character, conflict, and emotional tone.

It was developed as part of a CTO-level learning exercise to deepen understanding of **storytelling as a strategic communication tool**.

---

## 📁 Root Directory

| File / Folder          | Purpose                                                                                                          |
| -------------          | ---------------------------------------------------------------------------------------------------------------- |
| `storytelling.py`      | Main Streamlit app. Hosts the UI for video entry, executes the storytelling analysis pipeline.                   |
| `./`                   | Stores CSV exports generated after analyzing talks.                                                              |

---

## 📄 Key Scripts

### `storytelling.py`

* Accepts YouTube video links or IDs via a text input box.
* Fetches transcripts using `youtube-transcript-api`.
* Processes the transcript to:
  * Count emotional words
  * Identify storytelling patterns (protagonist, conflict, resolution)
  * Generate visuals (tag frequency, word cloud)
* Optionally integrates with GPT (if enabled) for deeper insights.
* Provides CSV download of structured analysis.
* Handles transcript fetching from YouTube based on video ID.
* Converts structured transcript into clean text.
* Implements tagging logic for storytelling attributes (e.g., “humor”, “personal story”).
* Contains prompt engineering for GPT calls to classify and summarize storytelling elements.

---
## Steps

1. Install the dependencies
```bash
pip install streamlit youtube-transcript-api pandas openai wordcloud
```

2. Execute the script
```bash
streamlit run storytelling.py
```

3. Access the app in the url shared by StreamLit

---

## 🧪 Dependencies

```bash
streamlit
openai
youtube-transcript-api
pandas
matplotlib
wordcloud
```

---

## 📤 Output Files

| File                             | Description                                   |
| ------------------------------   | --------------------------------------------- |
| `storytelling_analysis.csv`      | Summary of talk analysis in table form.       |

---
