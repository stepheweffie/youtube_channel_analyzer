import d6tflow
import argparse
from workflow import DownloadTranscriptTask
from googleapiclient.discovery import build

api_key = 'api_key'
yt_api_key = api_key
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
    for video_id in video_ids:
        download = DownloadTranscriptTask(video_id=video_id)
        try:
            d6tflow.run(download)
        except RuntimeError:
            print(video_id)


def main(channel_url):
    process_video_transcripts(channel_url)
    # get_video_ids(get_channel_id(channel_url))
    # get_latest_video(get_channel_id(channel_url))


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process transcripts for a YouTube channel')
    parser.add_argument('channel_url', type=str, help='The URL of the YouTube channel')
    args = parser.parse_args()
    main(args.channel_url)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
