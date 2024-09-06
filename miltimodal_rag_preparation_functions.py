import os
from pytube import YouTube
import gdown
import random
from yt_dlp import YoutubeDL
# from aichamptools.llms import OnReplicate
import subprocess
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json

load_dotenv(".env")



# def collect_video_youtube(url, collect_metadata=[], transcriber="replicate"):

#     tmp_media_folder = "tmp_media"


#     # Get video
#     #  direct download link from pytube

#     try:
#         print("trying to get video metadata from pytube...")

#         video = YouTube(url)
#         stream = video.streams.get_highest_resolution()
#         video_id = video.video_id
#         video_download_link = stream.url

#         print("success!\n")


#     # If Pytube fails,
#     #  try to download video with yt-dlp

#     except Exception as e:

#         print("failed to get video metadata from pytube, trying to download video with yt-dlp...")

#         ydl_opts = {
#             'format': 'bestaudio/best',
#             'postprocessors': [{
#                 'key': 'FFmpegExtractAudio',
#                 'preferredcodec': 'mp3',
#                 'preferredquality': '192',
#             }],
#             'outtmpl': f'{tmp_media_folder}%(id)s.%(ext)s',
#             'quiet': True
#         }
#         with YoutubeDL(ydl_opts) as ydl:
#             info_dict = ydl.extract_info(url, download=True)
#             video_download_link = info_dict['url']
#             video_id = info_dict['id']

#         print("success!\n")


#     # Transcribe with diarisation
#     #  using Replicate

#     if transcriber == "replicate":

#         print(f"transcribing with replicate (direct link: '{video_download_link}')...")

#         llm = OnReplicate(api_key=os.getenv("REPLICATE_API_TOKEN"), log_on=True)
#         transcription = llm.transcribe(link=video_download_link)

#         print("success!\n")


#     # Save transcription
#     #  to JSON file

#     os.makedirs('tmp', exist_ok=True)
#     json_path = os.path.join('tmp', f'video_{video_id}_transcript.json')
#     with open(json_path, 'w', encoding='utf-8') as f:
#         json.dump(transcription, f, ensure_ascii=False, indent=4)

#     print(f"Transcription saved to {json_path}")


#     return transcription


def extract_full_text_from_diarised_transcript(json_file_path, clean_file=False):
    """
    Extract full text from a diarised transcript JSON file generated by https://replicate.com/thomasmol/whisper-diarization.
    If clean_file is True, it assumes the file is a cleaned version containing only words and their timestamps.
    """

    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    full_text = ""
    if clean_file:
        for word_info in data:
            full_text += word_info['word'] + " "
    else:
        for segment in data['output']['segments']:
            full_text += segment['text'] + " "
    
    return full_text.strip()


def extract_only_word_ts_from_transcript(input_file, output_file):
    """
    Extract word timestamps from a diarised transcript JSON file generated by https://replicate.com/thomasmol/whisper-diarization (basically, generated a clean version containing only words and their timestamps).
    """

    with open(input_file, 'r') as file:
        data = json.load(file)
    
    word_timestamps = []
    
    for segment in data['output']['segments']:
        for word_info in segment['words']:
            word_timestamps.append({
                "start": word_info['start'],
                "end": word_info['end'],
                "word": word_info['word']
            })
    
    with open(output_file, 'w') as file:
        json.dump(word_timestamps, file, ensure_ascii=False, indent=4)


def find_phrase_timestamps(word_timestamps_file, phrase):
    with open(word_timestamps_file, 'r') as file:
        data = json.load(file)

    words = phrase.split()
    word_count = len(words)
    
    for i in range(len(data) - word_count + 1):
        match = True
        for j in range(word_count):
            if data[i + j]['word'] != words[j]:
                match = False
                break
        if match:
            start_timestamp = data[i]['start']
            end_timestamp = data[i + word_count - 1]['end']
            return start_timestamp, end_timestamp
    
    return None, None


def process_segments(segments_file, word_timestamps_file):
    with open(segments_file, 'r') as file:
        segments = json.load(file)
    
    for segment in segments:
        start_phrase = segment['start_phrase']
        finish_phrase = segment['finish_phrase']

        full_text = extract_full_text_from_diarised_transcript(word_timestamps_file, clean_file=True)
        
        start_timestamp, _ = find_phrase_timestamps(word_timestamps_file, start_phrase)
        _, end_timestamp = find_phrase_timestamps(word_timestamps_file, finish_phrase)

        start_index = full_text.find(start_phrase)
        finish_index = full_text.find(finish_phrase, start_index)
        
        if start_index != -1 and finish_index != -1:
            full_segment_text = full_text[start_index:finish_index + len(finish_phrase)]
        else:
            full_segment_text = None
        
        # Convert timestamps to HH:MM:SS format
        start_timestamp_hhmmss = convert_to_hhmmss(start_timestamp) if start_timestamp else None
        end_timestamp_hhmmss = convert_to_hhmmss(end_timestamp) if end_timestamp else None
        
        # Generate snake_case title for media field
        media_title = segment['name'].lower().replace(' ', '_') + '.png'
        
        # Update segment with required keys only
        segment_updated = {
            'name': segment['name'],
            'start_phrase': start_phrase,
            'finish_phrase': finish_phrase,
            'full_text': full_segment_text,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'start_timestamp_hhmmss': start_timestamp_hhmmss,
            'end_timestamp_hhmmss': end_timestamp_hhmmss,
            'media': media_title
        }

        # Replace the original segment with the updated one
        segments[segments.index(segment)] = segment_updated

    with open(segments_file, 'w') as file:
        json.dump(segments, file, indent=4)


def convert_to_hhmmss(seconds):
    if seconds is None:
        return None
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


from langchain_core.documents import Document


def segments_to_langchain_documents(segments_file):
    with open(segments_file, 'r') as file:
        segments = json.load(file)
    
    documents = []

    for segment in segments:
        document = Document(
            page_content=segment['full_text'],
            metadata={
                "name": segment.get('name', ""),
                "start_phrase": segment.get('start_phrase', ""),
                "finish_phrase": segment.get('finish_phrase', ""),
                "start_timestamp": segment.get('start_timestamp', ""),
                "end_timestamp": segment.get('end_timestamp', ""),
                "start_timestamp_hhmmss": segment.get('start_timestamp_hhmmss', ""),
                "end_timestamp_hhmmss": segment.get('end_timestamp_hhmmss', ""),
                "media": segment.get('media', ""),
                "media_description": segment.get('media_description', ""),
                "type": segment.get('type', "transcript_chunk")
            }
        )
        documents.append(document)

    print(f"first document: {documents[0]}")
    print(f"last document: {documents[-1]}")
    print(f"number of documents: {len(documents)}")

    return documents


