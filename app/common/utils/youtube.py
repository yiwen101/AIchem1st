from app.model.structs import ParquetFileRow, YoutubeVideoInfo
from pytubefix import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import re

def get_youtube_video_info(query_row: ParquetFileRow) -> YoutubeVideoInfo:
    youtube_url = query_row.youtube_url
    video_length = query_row.duration
    return _get_youtube_video_info(youtube_url, video_length)

def _get_video_id(url):
    """Extract the video ID from a YouTube URL."""
    # This regex handles various YouTube URL formats
    youtube_regex = r'(youtu\.be\/|youtube\.com\/(watch\?(.*&)?v=|(embed|v|shorts)\/))([^?&"\'>]+)'
    match = re.search(youtube_regex, url)
    if match:
        return match.group(5)
    return None

def _convert_to_watch_url(url):
    """Convert any YouTube URL to the standard watch format."""
    video_id = _get_video_id(url)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

def _get_youtube_video_info(url: str, video_length: str) -> YoutubeVideoInfo:
    """Get title, description and transcript of a YouTube video."""
    title = ""
    description = ""
    transcript = ""
    try:
        # Extract video ID
        video_id = _get_video_id(url)
        if not video_id:
            return YoutubeVideoInfo(title=title, description=description, transcript=transcript, video_length=video_length)
        
        # Convert to standard watch URL format (helps with Shorts)
        watch_url = _convert_to_watch_url(url)
        print("watch_url: ", watch_url)
        
        # Create YouTube object with additional arguments to avoid errors
        yt = YouTube(watch_url)
        print("yt: ", yt)
        
        # Get title and description
        title = yt.title
        print("title: ", title)
        description = yt.description
        print("description: ", description)
        
        # Get transcript
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            print("transcript_list: ", transcript_list)
            transcript = " ".join([item['text'] for item in transcript_list])
        except Exception as e:
            transcript = "The video have no transcript."
        
        info = YoutubeVideoInfo(title=title, description=description, transcript=transcript, video_length=video_length)
        return info
    except Exception as e:
        return YoutubeVideoInfo(title=title, description=description, transcript=transcript, video_length=video_length)