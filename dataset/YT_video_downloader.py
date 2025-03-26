import sys
import os
from multiprocessing.pool import ThreadPool
import csv
from yt_dlp import YoutubeDL
import ffmpeg
import subprocess
# CSV file format: 'yt_video_id, file_name, start_time (in seconds), end_time (in seconds)'

class VidInfo:
    def __init__(self, yt_id, file_name, start_time, end_time, trash, outdir):
        self.yt_id = yt_id.strip()
        if ':' not in start_time:
          start_time = start_time + ':' + '0' + '0'
        self.start_time = start_time.strip()
        if ':' not in end_time:
          end_time = end_time + ':' + '0' + '0'
        self.end_time = end_time.strip()

        self.out_filename = os.path.join(outdir, file_name.strip() + '.mp4')
   

def download(vidinfo):
    yt_url = f"https://www.youtube.com/watch?v={vidinfo.yt_id}"
    yt_url = yt_url.strip().replace('\ufeff', '')

    temp_filename = vidinfo.out_filename.replace('.mp4', '_full.mp4')

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': temp_filename,
        'quiet': False,
        'merge_output_format': 'mp4'
    }

    os.makedirs(os.path.dirname(vidinfo.out_filename), exist_ok=True)

    with YoutubeDL(ydl_opts) as ydl:
        print(f"Downloading full video: {yt_url}")
        ydl.download([yt_url])

    # Recortar con ffmpeg
    ffmpeg_cmd = [
        "ffmpeg", "-i", temp_filename,
        "-ss", vidinfo.start_time,
        "-to", vidinfo.end_time,
        "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k",  # Asegura que el audio esté presente
        "-strict", "experimental",
        vidinfo.out_filename
    ]

    print("Recortando video con ffmpeg...")
    subprocess.run(ffmpeg_cmd, check=True)

    # Opcional: eliminar el video completo si ya no es necesario
    os.remove(temp_filename)

    return f"{vidinfo.yt_id}, DONE!"

if __name__ == '__main__':
    csv_file = "C:/Users/victo/deceptionDetector/dataset/DOLOS/dolos_timestamps.csv"
    out_dir = "C:/Users/victo/deceptionDetector/dataset/videos"

    os.makedirs(out_dir, exist_ok=True)

    vidinfos = []
    cnt=_cnt=0 #1680
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 5 and cnt<_cnt+30:
                vidinfos.append(VidInfo(*row, out_dir))
                cnt += 1

    with open('bad_files_a.txt', 'w', encoding='utf-8') as bad_files:
        results = ThreadPool(20).imap_unordered(download, vidinfos)
        for cnt, r in enumerate(results, start=1):
            print(f"{cnt}/{len(vidinfos)} {r}")
            if 'ERROR' in r:
                bad_files.write(r + '\n')