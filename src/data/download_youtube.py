import os
import argparse
import yt_dlp

def download_video(url: str, output_path: str, max_height: int = 720):
    """
    Downloads a YouTube video (video stream only, no audio) with a capped resolution
    to avoid huge files.
    """
    # Create the output directory if it does not exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Force the H.264 (avc1) codec so that OpenCV can always read the file.
    # Ask for the best mp4 format whose height is <= max_height,
    # and do not download the audio.
    ydl_opts = {
        'format': f'bestvideo[vcodec^=avc1][height<={max_height}][ext=mp4]/best[vcodec^=avc1][height<={max_height}][ext=mp4]/bestvideo[height<={max_height}][ext=mp4]',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
    }
    
    print(f"Downloading video from {url} (max resolution: {max_height}p)...")
    print(f"Output file: {output_path}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("\nDownload completed successfully!")
    except Exception as e:
        print(f"\nError while downloading: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a YouTube video locally")
    parser.add_argument("--url", type=str, required=True, help="YouTube video URL")
    parser.add_argument("--out", type=str, default="./data/source_video.mp4", help="Output file path (e.g. ./data/video.mp4)")
    parser.add_argument("--height", type=int, default=720, help="Maximum resolution (e.g. 720 for 720p, 480 for 480p)")
    
    args = parser.parse_args()
    
    download_video(args.url, args.out, max_height=args.height)
