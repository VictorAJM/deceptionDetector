import sys
import os
from multiprocessing.pool import ThreadPool
import csv
from yt_dlp import YoutubeDL
import ffmpeg
import subprocess
from collections import defaultdict

class VidClip:
    def __init__(self, file_name, start_time, end_time, outdir):
        self.file_name = file_name.strip()
        self.start_time = self._format_time(start_time.strip())
        self.end_time = self._format_time(end_time.strip())
        self.out_filename = os.path.join(outdir, self.file_name + '.mp4')

    def _format_time(self, time_str):
        return time_str if ':' in time_str else f"{time_str}:00"


def download_video(yt_id, temp_filename):
    yt_url = f"https://www.youtube.com/watch?v={yt_id.strip()}"
    yt_url = yt_url.strip().replace('\ufeff', '')
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': temp_filename,
        'quiet': False,
        'merge_output_format': 'mp4'
    }
    
    if not os.path.exists(temp_filename):
        with YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading full video: {yt_url}")
            ydl.download([yt_url])
    else:
        print(f"Skipping download, file already exists: {temp_filename}")


def extract_clip(temp_filename, clip):
    ffmpeg_cmd = [
        "ffmpeg", "-i", temp_filename,
        "-ss", clip.start_time,
        "-to", clip.end_time,
        "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k", 
        "-strict", "experimental",
        clip.out_filename
    ]
    
    print(f"Extracting clip: {clip.out_filename}")
    subprocess.run(ffmpeg_cmd, check=True)


def process_video(yt_id, clips, outdir):
    temp_filename = os.path.join(outdir, f"{yt_id}_full.mp4")
    download_video(yt_id, temp_filename)
    
    for clip in clips:
        extract_clip(temp_filename, clip)
    
    os.remove(temp_filename)  # Remove full video after processing all clips


if __name__ == '__main__':
    csv_file = "C:/Users/victo/deceptionDetector/dataset/DOLOS/dolos_timestamps.csv"
    out_dir = "C:/Users/victo/deceptionDetector/dataset/videos"
    os.makedirs(out_dir, exist_ok=True)
    
    video_clips = defaultdict(list)
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 5:
                yt_id, file_name, start_time, end_time, _ = row
                video_clips[yt_id].append(VidClip(file_name, start_time, end_time, out_dir))
    video_clips = list(video_clips.items())[35:40]
    with open('bad_files_a.txt', 'w', encoding='utf-8') as bad_files:
        results = ThreadPool(5).imap_unordered(lambda item: process_video(item[0], item[1], out_dir), video_clips)
        for cnt, r in enumerate(results, start=1):
            print(f"{cnt}/{len(video_clips)} processed")