from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
yt_api_key = os.getenv('YOUTUBE_API_KEY')