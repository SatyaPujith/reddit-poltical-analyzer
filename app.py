from flask import Flask, render_template, request
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import os

app = Flask(__name__)

# Initialize Reddit API
reddit = praw.Reddit(
    client_id="pwd19Vei5UGVQX49CBBO4A",
    client_secret="UDwq-jZyVQfkWsH4AnNeBFK4pegMnw",
    user_agent="political_analysis_bot pujth by /u/Fragrant_Ad_4239"
)

analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|\#", "", text)
    return text.strip()

def fetch_reddit_data(query, subreddit="politics", limit=200):
    subreddit = reddit.subreddit(subreddit)
    posts = subreddit.search(query, limit=limit)
    data = []
    try:
        for i, post in enumerate(posts, 1):
            text = clean_text(post.title + " " + post.selftext)
            sentiment = analyzer.polarity_scores(text)["compound"]
            data.append({
                "id": i,
                "text": text,
                "sentiment": sentiment,
                "created": pd.to_datetime(post.created_utc, unit='s'),
                "score": post.score
            })
    except Exception as e:
        print(f"Error fetching data: {e}")
    df = pd.DataFrame(data)
    if not df.empty:
        df.to_csv(f"{query}_reddit_data.csv", index=False)
    return df

def analyze_data(df, query):
    if df.empty:
        return df, "neutral", {}, {}, 0, 0, 0

    filtered_df = df[df['text'].str.contains(query, case=False, na=False)]
    if filtered_df.empty:
        filtered_df = df

    avg_score = filtered_df['sentiment'].mean()
    sentiment = "positive" if avg_score > 0.05 else "negative" if avg_score < -0.05 else "neutral"
    pos_count = len(filtered_df[filtered_df['sentiment'] > 0.05])
    neg_count = len(filtered_df[filtered_df['sentiment'] < -0.05])
    neu_count = len(filtered_df[(filtered_df['sentiment'] >= -0.05) & (filtered_df['sentiment'] <= 0.05)])

    all_text = " ".join(filtered_df['text'].str.lower())
    words = [w for w in all_text.split() if len(w) > 3]
    word_freq = dict(Counter(words).most_common(5))

    cluster_words = {}
    if len(filtered_df) >= 3:
        vectorizer = CountVectorizer(max_features=100, stop_words='english')
        X = vectorizer.fit_transform(filtered_df['text']).toarray()
        n_clusters = min(3, len(filtered_df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        filtered_df['cluster'] = clusters
        for cluster in range(n_clusters):
            cluster_text = " ".join(filtered_df[filtered_df['cluster'] == cluster]['text'].str.lower())
            cluster_words[cluster] = [w for w, c in Counter(cluster_text.split()).most_common(3) if len(w) > 3]
    else:
        filtered_df['cluster'] = 0
        cluster_words[0] = list(word_freq.keys())[:3]

    return filtered_df, sentiment, word_freq, cluster_words, pos_count, neg_count, neu_count

def visualize_data(df, query, sentiment, word_freq, cluster_words, pos_count, neg_count, neu_count):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(df['created'], df['sentiment'], marker='o')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f"Sentiment Over Time for '{query}'")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [pos_count, neg_count, neu_count]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Sentiment Distribution")

    plt.subplot(2, 2, 3)
    words, freqs = zip(*word_freq.items())
    plt.bar(words, freqs, color='skyblue')
    plt.title("Top Words")
    plt.xlabel("Words")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 4)
    cluster_sizes = df['cluster'].value_counts()
    plt.bar(cluster_sizes.index, cluster_sizes.values, color='lightgreen')
    plt.title("Cluster Sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Post Count")

    plt.tight_layout()
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/plot.png")
    plt.close()

@app.route("/", methods=["GET", "POST"])
def home():
    response = None
    plot_url = None
    table_data = None
    if request.method == "POST":
        question = request.form.get("question")
        subreddit = "politics"

        if ":" in question:
            parts = question.split(":")
            subreddit = parts[0].strip().replace("r/", "")
            question = parts[1].strip()

        df = fetch_reddit_data(question, subreddit)
        if df.empty:
            response = f"No data found for '{question}' in r/{subreddit}."
        else:
            df, sentiment, word_freq, cluster_words, pos_count, neg_count, neu_count = analyze_data(df, question)
            visualize_data(df, question, sentiment, word_freq, cluster_words, pos_count, neg_count, neu_count)

            trends = ", ".join([f"{word} ({count}x)" for word, count in word_freq.items()])
            cluster_summary = "\n".join([f"Cluster {k}: {', '.join(v)}" for k, v in cluster_words.items()])
            response = (f"Sentiment for '{question}' in r/{subreddit} is **{sentiment}**.\n"
                        f"Positive: {pos_count}, Negative: {neg_count}, Neutral: {neu_count}.\n\n"
                        f"Top trends: {trends}\n\nCluster Topics:\n{cluster_summary}")
            plot_url = "/static/plot.png"
            table_data = df[['id', 'text', 'sentiment', 'score', 'created', 'cluster']].to_dict(orient="records")

    return render_template("index.html", response=response, plot_url=plot_url, table_data=table_data)

if __name__ == "__main__":
    app.run(debug=True)
