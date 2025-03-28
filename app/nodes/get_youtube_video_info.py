from app.model.state import VideoAgentState
from app.model.structs import YoutubeVideoInfo
from pytubefix import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import re

def get_youtube_video_info(state: VideoAgentState) -> VideoAgentState:
    youtube_url = state["query"].youtube_url
    info = get_youtube_video_info(youtube_url)
    return {"video_info": info}




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
    video_id = get_video_id(url)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

def _get_youtube_video_info(url: str) -> YoutubeVideoInfo:
    """Get title, description and transcript of a YouTube video."""
    try:
        # Extract video ID
        video_id = _get_video_id(url)
        if not video_id:
            return "Invalid YouTube URL"
        
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
            transcript = " ".join([item['text'] for item in transcript_list])
        except Exception as e:
            transcript = f"Transcript unavailable: {str(e)}"
        
        info = YoutubeVideoInfo(title=title, description=description, transcript=transcript)
        return info
    except Exception as e:
        return f"An error occurred: {str(e)}"