import os
import csv
import subprocess
import time
from collections import defaultdict
from multiprocessing.pool import ThreadPool

from yt_dlp import YoutubeDL


class VidClip:
    """Representa un clip individual dentro de un video de YouTube."""
    def __init__(self, file_name, start_time, end_time, outdir):
        self.file_name = file_name.strip()
        self.start_time = self._format_time(start_time.strip())
        self.end_time = self._format_time(end_time.strip())
        self.out_filename = os.path.join(outdir, f"{self.file_name}.mp4")

    @staticmethod
    def _format_time(time_str):
        # Asegura formato HH:MM:SS
        return time_str if ':' in time_str else f"{time_str}:00"


class VideoProcessor:
    """Encapsula la lógica de descarga, extracción y reporte de videos."""
    def __init__(self, csv_path, out_dir, thread_count=5, cookies_path=None):
        self.csv_path = csv_path
        self.out_dir = out_dir
        self.thread_count = thread_count
        self.cookies_path = cookies_path
        os.makedirs(self.out_dir, exist_ok=True)

    def load_clips(self):
        """Lee el CSV y agrupa los clips por yt_id."""
        clips_map = defaultdict(list)
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 5:
                    yt_id, fn, st, et, _ = row
                    clips_map[yt_id].append(VidClip(fn, st, et, self.out_dir))
        return clips_map

    def count_videos(self, clips_map):
        """Cuenta e imprime la cantidad de videos únicos."""
        total = len(clips_map)
        print(f"Total de videos únicos encontrados: {total}")
        return total

    def download_video(self, yt_id, temp_filepath):
        """Descarga el video completo si no existe ya, usando cookies si se proporcionan."""
        url = f"https://www.youtube.com/watch?v={yt_id}"
        url = url.strip().replace('\ufeff', '')
        if os.path.exists(temp_filepath):
            return True

        opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': temp_filepath,
            'quiet': True,
            'merge_output_format': 'mp4'
        }
        if self.cookies_path:
            opts['cookiefile'] = self.cookies_path

        try:
            with YoutubeDL(opts) as ydl:
                ydl.download([url])
            if os.path.getsize(temp_filepath) == 0:
                raise RuntimeError("Archivo vacío tras descarga")
            return True
        except Exception as e:
            print(f"[ERROR descarga {yt_id}] {e}")
            return False

    @staticmethod
    def extract_clips(temp_filepath, clips):
        """Extrae todos los clips definidos de un archivo de video."""
        for clip in clips:
            cmd = [
                "ffmpeg", "-i", temp_filepath,
                "-ss", clip.start_time, "-to", clip.end_time,
                "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k",
                "-strict", "experimental",
                clip.out_filename
            ]
            subprocess.run(cmd, check=True)

    def process_single(self, args):
        """Worker: descarga y extrae; devuelve yt_id si falla."""
        yt_id, clips = args
        temp_file = os.path.join(self.out_dir, f"{yt_id}_full.mp4")

        if not self.download_video(yt_id, temp_file):
            return yt_id

        try:
            self.extract_clips(temp_file, clips)
        except Exception as e:
            print(f"[ERROR extracción {yt_id}] {e}")
            return yt_id
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        return None

    def run(self, subset=None):
        """
        Ejecuta el procesamiento de videos en paralelo.
        Si `subset` es una tupla (start, end), procesa sólo ese rango de videos.
        """
        clips_map = self.load_clips()
        self.count_videos(clips_map)

        items = list(clips_map.items())
        if subset:
            start, end = subset
            items = items[start:end]

        bad_list = []
        with ThreadPool(self.thread_count) as pool:
            for idx, res in enumerate(pool.imap_unordered(self.process_single, items), 1):
                if res:
                    bad_list.append(res)
                print(f"Procesados: {idx}/{len(items)}", end='\r')

        bad_path = os.path.join(self.out_dir, 'bad_videos.txt')
        with open(bad_path, 'w', encoding='utf-8') as bf:
            for yt in bad_list:
                bf.write(f"{yt}\n")

        print(f"\nFallidos: {len(bad_list)} videos (ver {bad_path})")
        return bad_list

    def retry_failed(self, failed_ids, max_attempts=3):
        """
        Reintenta la descarga y extracción de los IDs fallidos hasta max_attempts veces.
        Devuelve la lista final de IDs que siguen fallando.
        """
        print(f"Reintentando {len(failed_ids)} IDs hasta {max_attempts} veces...")
        attempt = 1
        still_bad = set(failed_ids)
        clips_map = self.load_clips()
        while attempt <= max_attempts and still_bad:
            print(f"-- Intento {attempt}")
            for yt_id in list(still_bad):
                clips = clips_map.get(yt_id, [])
                res = self.process_single((yt_id, clips))
                if res:
                    still_bad.add(res)
                else:
                    still_bad.discard(yt_id)
            print(f"Quedan {len(still_bad)} IDs tras intento {attempt}")
            attempt += 1
        return list(still_bad)

    def check_downloads(self, delay_between=0.0):
        """
        Intenta descargar cada video completo y escribe en un archivo los IDs que fallen.
        No extrae clips ni procesa nada más.
        """
        clips_map = self.load_clips()
        bad_ids = []

        for idx, yt_id in enumerate(clips_map.keys(), 1):
            temp_file = os.path.join(self.out_dir, f"{yt_id}_full.mp4")
            success = self.download_video(yt_id, temp_file)
            if not success:
                bad_ids.append(yt_id)
            else:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

            print(f"[{idx}/{len(clips_map)}] Comprobado: {yt_id} -> {'OK' if success else 'FAIL'}", end='\r')
            if delay_between:
                time.sleep(delay_between)

        bad_path = os.path.join(self.out_dir, 'bad_downloads.txt')
        with open(bad_path, 'w', encoding='utf-8') as f:
            for yt in bad_ids:
                f.write(f"{yt}\n")

        print(f"\nResultados escritos en: {bad_path}")
        print(f"Total intentos: {len(clips_map)}, Fallos: {len(bad_ids)}")


if __name__ == '__main__':
    CSV_PATH = "dataset/DOLOS/dolos_timestamps.csv"
    OUT_DIR = "dataset/videos"
    COOKIES = os.path.expanduser('~/.cookies/youtube.txt')  # archivo de cookies opcional

    processor = VideoProcessor(CSV_PATH, OUT_DIR, thread_count=5, cookies_path=COOKIES)
    # Leer IDs fallidos desde el archivo bad_videos.txt
    bad_path = os.path.join(OUT_DIR, 'bad_videos.txt')
    failed = []
    if os.path.exists(bad_path):
        with open(bad_path, 'r', encoding='utf-8') as bf:
            failed = [line.strip() for line in bf if line.strip()]
    # Reintentar los fallidos si existen
    if failed:
        final_bad = processor.retry_failed(failed, max_attempts=3)
        if final_bad:
            print(f"IDs que siguen fallando: {final_bad}")
