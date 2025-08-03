# In utils/youtube_utils.py

import yt_dlp
import logging
import re
from youtube_transcript_api import YouTubeTranscriptApi
import time
from bs4 import BeautifulSoup
import concurrent.futures
import requests
from typing import List, Dict, Optional
from datetime import datetime
from urllib.parse import urlparse, urlunparse
log = logging.getLogger(__name__)

def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en','hi'])
    return "\n".join([segment['text'] for segment in transcript])

    
def is_youtube_video_url(url: str) -> bool:
    """
    Checks if a URL is a YouTube video URL (watch or youtu.be).
    """
    # Regex to match standard watch URLs and shortened youtu.be URLs
    video_pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)'
    return re.match(video_pattern, url) is not None

def clean_youtube_url(url):
    """
    Remove query parameters (like 'si') from a YouTube URL.
    
    Args:
        url (str): The YouTube URL with query parameters
        
    Returns:
        str: The clean YouTube URL without query parameters
    """
    parsed = urlparse(url)
    
    # Reconstruct URL without query parameters
    clean_url = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        '',  # params (empty)
        '',  # query (empty)
        ''   # fragment (empty)
    ))
    
    return clean_url

# This function is correct and does not need changes.
def extract_channel_videos(channel_url, max_videos=50):
    """Extract video URLs, high-quality thumbnail, and subscriber count from a YouTube channel"""
    channel_url = clean_youtube_url(channel_url)
    try:
        ydl_opts = {
            'quiet': False,
            'verbose': False,
            'no_warnings': True,
            'extract_flat': True,
            'playlistend': max_videos,
            'socket_timeout': 30,
            'cookiefile': 'cookies.txt',
        }
        
        # Construct the /videos URL
        if '/channel/' in channel_url or '/@' in channel_url or '/c/' in channel_url or '/user/' in channel_url:
            playlist_url = channel_url.split('/videos')[0] + '/videos'
        else:
            playlist_url = channel_url + '/videos'
        
        print(f"Fetching video list from: {playlist_url} with verbose output...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Invoking yt-dlp to extract info...")
            info = ydl.extract_info(playlist_url, download=False)
            print("yt-dlp extract_info call finished.")
            
            if 'entries' in info:
                video_urls = [f"https://www.youtube.com/watch?v={entry['id']}" for entry in info['entries'][:max_videos] if entry and 'id' in entry]
                
                channel_thumbnail = info.get('thumbnails', [{}])[-1].get('url', info.get('uploader_avatar', ''))
                
                # --- START: MODIFICATION ---
                # Get subscriber count, default to 0 if not found
                subscriber_count = info.get('channel_follower_count', 0)
                print(f"Found subscriber count: {subscriber_count}")
                
                return video_urls, channel_thumbnail, subscriber_count
                # --- END: MODIFICATION ---
            
    except Exception as e:
        print(f"Error extracting videos: {e}")
    
    # Return default values on failure
    return [], '', 0

# This function is correct and does not need changes.
def _parse_vtt_transcript(vtt_content: str) -> str:
    """
    A helper function to parse a .vtt file content and extract clean transcript text.
    """
    lines = vtt_content.strip().split('\n')
    clean_lines = []
    for line in lines:
        if '-->' in line or line.strip().upper() in ['WEBVTT', 'KIND: CAPTIONS', 'LANGUAGE: EN']:
            continue
        line = re.sub(r'<[^>]+>', '', line)
        if line.strip():
            clean_lines.append(line.strip())
    
    unique_lines = []
    last_line = None
    for line in clean_lines:
        if line != last_line:
            unique_lines.append(line)
            last_line = line
            
    return " ".join(unique_lines)

# This is the primary function we are modifying.
# In utils/youtube_utils.py


def get_video_transcripts(video_urls: List[str], max_videos: int = 50, progress_callback=None) -> List[Dict]:
    """
    Processes a list of video URLs in parallel. It fetches metadata using yt-dlp
    and scrapes the transcript from youtubetotranscript.com.
    """
    # --- TIMER START ---
    start_time = time.perf_counter()
    # --- TIMER END ---

    if not isinstance(video_urls, list):
        log.error("Data type error: expects a list of URLs.")
        raise TypeError("get_video_transcripts was called with an incorrect data type.")

    urls_to_process = video_urls[:max_videos]
    print(f"Fetching transcripts for {len(urls_to_process)} videos using scraper method...")

    def process_url(url: str) -> Optional[Dict]:
        """
        Worker function that processes a single URL.
        1. Gets metadata via yt-dlp.
        2. Scrapes transcript from external site.
        """
        try:
            # --- STEP 1: Get video metadata and ID using yt-dlp ---
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=False)
            
            video_id = info_dict.get('id')
            if not video_id:
                log.warning(f"Could not extract video ID for URL: {url}")
                return None

            # --- STEP 2: Scrape the transcript using the video ID ---
            transcript_text = None
            try:
                transcript_url = f"https://youtubetotranscript.com/transcript?v={video_id}"
                response = requests.get(transcript_url, timeout=30)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                
                transcript_container = soup.find('div', class_='-mt-4') or soup.find('article')
                if not transcript_container:
                    log.warning(f"Transcript container not found for video ID {video_id}. Site layout may have changed.")
                    return None
                
                filter_keywords = [
                    "SponsorBlock", "Recapio", "Author :", "free prompts", "on steroids", "magic words"
                ]

                paragraphs = transcript_container.find_all('p')
                transcript_lines = [
                    p.get_text(" ", strip=True)
                    for p in paragraphs
                    if p.get_text(" ", strip=True) and not any(kw in p.get_text() for kw in filter_keywords)
                ]

                if not transcript_lines:
                    log.warning(f"No valid transcript text found after scraping for video ID {video_id}.")
                    return None

                transcript_text = "\n".join(transcript_lines)

            except Exception as e:
                print(f"first method failed because: {e}")
                try:
                    print("using 2nd method")
                    transcript_text = get_transcript(video_id)
                except Exception as e:
                    log.error(f"Failed to scrape transcript for {url}: {e}", exc_info=False)
                    return None

            # --- STEP 3: Combine metadata and transcript into a final dictionary ---
            upload_date_str = info_dict.get('upload_date')
            formatted_date = None
            if upload_date_str:
                try:
                    dt_object = datetime.strptime(upload_date_str, '%Y%m%d')
                    formatted_date = dt_object.strftime('%Y-%m-%d')
                except ValueError:
                    formatted_date = upload_date_str
            
            video_data = {
                'video_id': info_dict.get('id'),
                'url': info_dict.get('webpage_url'),
                'title': info_dict.get('title'),
                'uploader': info_dict.get('uploader'),
                'description': info_dict.get('description'),
                'duration': info_dict.get('duration'),
                'upload_date': formatted_date,
                'transcript': transcript_text
            }
            print(f"✅ Successfully processed transcript for: {video_data['title'][:50]}...")
            return video_data

        except Exception as e:
            log.error(f"❌ Failed to process video {url}: {e}", exc_info=True)
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_url = {executor.submit(process_url, url): url for url in urls_to_process}
        results = []
        
        total_videos = len(future_to_url)
        for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                url_for_exc = future_to_url[future]
                log.error(f"URL {url_for_exc} generated an exception: {exc}")
            
            if progress_callback:
                progress_callback(i + 1, total_videos)

    transcripts_data = [res for res in results if res is not None]
    if not transcripts_data:
        log.warning("Could not extract any transcripts from the provided video URLs.")

    # --- TIMER START ---
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"\n[PERFORMANCE] Transcript extraction for {len(urls_to_process)} videos completed in {duration:.2f} seconds.\n")
    # --- TIMER END ---
    
    return transcripts_data


# MODIFIED TEST SCRIPT to show the upload date
if __name__ == '__main__':
    channel_url_input = input("Please enter the YouTube channel URL to test: ")
    
    if channel_url_input:
        print(f"\n--- Step 1: Extracting the first 5 videos from: {channel_url_input} ---\n")
        
        video_urls, thumbnail_url = extract_channel_videos(channel_url_input, max_videos=50)
        
        if video_urls:
            print(f"--- SUCCESS: Found {len(video_urls)} videos ---")
            print(f"Channel Thumbnail URL: {thumbnail_url}")
            for i, url in enumerate(video_urls):
                print(f"  {i+1}. {url}")
            
            print(f"\n--- Step 2: Fetching transcripts for the {len(video_urls)} extracted videos... ---\n")
            
            transcripts_data = get_video_transcripts(video_urls)
            
            if transcripts_data:
                print("--- FINAL SUCCESS: All transcripts received ---")
                for i, transcript in enumerate(transcripts_data):
                    transcript_snippet = transcript.get('transcript', '')[:150] + '...'
                    # --- START FIX ---
                    # Also print the upload date to verify the fix
                    upload_date = transcript.get('upload_date', 'N/A')
                    print(f"\n  Video {i+1}: {transcript.get('title')}")
                    print(f"  Upload Date: {upload_date}")
                    print(f"  Transcript Snippet: {transcript_snippet}")
                    # --- END FIX ---
            else:
                print("\n--- Test FAILED at Step 2: Could not fetch any transcripts. ---")

        else:
            print("\n--- Test FAILED at Step 1: Could not find any videos for that channel. ---")
    else:
        print("No URL entered. Exiting test.")