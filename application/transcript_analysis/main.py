# main.py
import d6tflow
import argparse
from workflow import NLPAnalysisTask, plot_word_histogram
from googleapiclient.discovery import build
from app import yt_api_key
from collections import Counter


youtube = build("youtube", "v3", developerKey=yt_api_key)


def get_channel_id(channel_url):
    channel_id = channel_url.split("/")[-1]
    return channel_id


def get_video_ids(channel_id, max_results=50):
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=max_results,
        order="date",
        type="video"
    )
    response = request.execute()
    video_ids = [item["id"]["videoId"] for item in response["items"]]
    return video_ids


def get_latest_video(channel_id):
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        order="date",
        maxResults=1,
        type="video",
    )
    response = request.execute()
    if response["items"]:
        video = response["items"][0]
        return video["id"]["videoId"], video["snippet"]["title"]
    return None


def process_video_transcripts(channel_url):
    channel_id = get_channel_id(channel_url)
    video_ids = get_video_ids(channel_id)
    processed_texts = {}

    for video_id in video_ids:
        nlp_task = NLPAnalysisTask(video_id=video_id)
        d6tflow.run(nlp_task)

        processed_text = nlp_task.output().load()
        print(f"Processed text for video ID {video_id}")

        # Perform a simple word frequency analysis
        word_frequencies = Counter()
        for video_id, processed_text in processed_texts.items():
            word_frequencies.update(processed_text)

        # Get the most common words
        most_common_words = word_frequencies.most_common(10)
        print("Most common words:")
        for word, count in most_common_words:
            print(f"{word}: {count}")
        # Plot the histogram
        plot_word_histogram(most_common_words)


def main(channel_url):
    process_video_transcripts(channel_url)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process transcripts for a YouTube channel')
    parser.add_argument('channel_url', type=str, help='The URL of the YouTube channel')
    args = parser.parse_args()
    main(args.channel_url)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
