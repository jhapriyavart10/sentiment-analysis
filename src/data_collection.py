"""
data_collection.py
- Fetch YouTube comments using API (if API key provided)
- Fallback: Load comments from CSV
"""
import os
import pandas as pd
from googleapiclient.discovery import build

def fetch_youtube_comments(api_key, video_id, max_comments=100):
    comments = []
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            textFormat='plainText'
        )
        response = request.execute()
        while response and len(comments) < max_comments:
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
                if len(comments) >= max_comments:
                    break
            if 'nextPageToken' in response:
                request = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100,
                    pageToken=response['nextPageToken'],
                    textFormat='plainText'
                )
                response = request.execute()
            else:
                break
    except Exception as e:
        print(f"Error fetching comments: {e}")
    return comments

def save_comments_to_csv(comments, path):
    df = pd.DataFrame({'comment': comments})
    df.to_csv(path, index=False)
    print(f"Saved {len(comments)} comments to {path}")

def load_comments_csv(path):
    return pd.read_csv(path)

if __name__ == "__main__":
    # Set your API key and video ID here
    API_KEY = os.getenv('YOUTUBE_API_KEY', '')
    VIDEO_ID = 'mjBym9uKth4'
    if API_KEY and VIDEO_ID:
        comments = fetch_youtube_comments(API_KEY, VIDEO_ID, max_comments=200)
        save_comments_to_csv(comments, 'data/comments.csv')
    else:
        print("No API key or video ID provided. Using sample CSV.")
        df = load_comments_csv('data/comments.csv')
        print(df.head())
