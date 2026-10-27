import os
import uuid
import subprocess
import whisper
import torch
import yt_dlp
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from urllib.parse import quote

app = Flask(__name__)

# Create a temporary folder to hold uploaded/downloaded files
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- THE SPEED FIX: Force the M3 GPU (MPS) ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading Whisper model on {device.upper()}...")
model = whisper.load_model("base", device=device) 
print("Model loaded! Ready to transcribe.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    youtube_url = request.form.get('youtube_url', '').strip()
    file = request.files.get('audio_file')
    
    temp_filepath = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4()}")
    actual_audio_path = None

    # 1. Determine the source of the audio
    if youtube_url: # This works on any video-sharing platform as well
        print(f"Downloading audio from YouTube...")
        
        # This function updates the terminal with download progress
        def progress_hook(d):
            if d['status'] == 'downloading':
                # \r moves the cursor to the start of the line to update it
                percent = d.get('_percent_str', 'Unknown')
                speed = d.get('_speed_str', 'Unknown')
                print(f"\rDownloading: {percent} | Speed: {speed}", end='', flush=True)
            elif d['status'] == 'finished':
                print("\nDownload complete! Extracting audio...")

        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': temp_filepath, 
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'progress_hooks': [progress_hook], # <--- This adds the progress bar!
                'no_warnings': True # Hides the Python 3.10 warning
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            actual_audio_path = temp_filepath + '.mp3'
        except Exception as e:
            return f"Error downloading YouTube video: {str(e)}"
            
    elif file and file.filename != '':
        # Handle standard file upload
        temp_filename = f"{temp_filepath}_{secure_filename(file.filename)}"
        actual_audio_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        file.save(actual_audio_path)
    else:
        return "Error: Please provide either a YouTube URL or upload an audio file."

    # 2. Handle the save directory
    save_dir = request.form.get('save_path', '').strip()
    if not save_dir:
        save_dir = os.path.expanduser("~/Desktop")
    else:
        save_dir = os.path.expanduser(save_dir)

    if not os.path.isdir(save_dir):
        return f"Error: The folder '{save_dir}' does not exist. Please check the path."

    try:
        # 3. Transcribe the audio
        print(f"Transcribing... (Check terminal for progress)")
        result = model.transcribe(actual_audio_path, verbose=True) 

        # 4. Save the text file with timestamps
        source_name = "youtube_transcript" if youtube_url else os.path.splitext(secure_filename(file.filename))[0]
        output_filename = f"{source_name}_transcript.txt"
        output_filepath = os.path.join(save_dir, output_filename)

        with open(output_filepath, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                start_time = format_timestamp(segment["start"])
                text = segment["text"].strip()
                f.write(f"[{start_time}] {text}\n\n")

        # 5. Create secure links
        folder_link = f"/open_folder?path={quote(save_dir)}"
        file_link = f"/open_file?path={quote(output_filepath)}"

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; text-align: center; }}
                .btn {{ display: inline-block; background: #007aff; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; margin: 10px; font-weight: 500; }}
                .btn-secondary {{ background: #f0f0f0; color: #333; }}
                .btn:hover {{ opacity: 0.9; }}
                a {{ color: #007aff; text-decoration: none; }}
            </style>
        </head>
        <body>
            <h2>Success!</h2>
            <p>Transcription saved as:</p>
            <p><strong>{output_filename}</strong></p>
            
            <div style="margin-top: 30px;">
                <a href="{folder_link}" class="btn">📁 Open Folder in Finder</a>
                <a href="{file_link}" class="btn btn-secondary">📄 Open Text File</a>
            </div>

            <br><br>
            <a href="/">← Transcribe another file</a>
        </body>
        </html>
        """

    except Exception as e:
        return f"Error during transcription: {str(e)}"
    
    finally:
        # 6. Clean up temporary files
        if actual_audio_path and os.path.exists(actual_audio_path):
            os.remove(actual_audio_path)

# --- ROUTES TO OPEN FILES/FOLDERS ---
@app.route('/open_folder')
def open_folder():
    path = request.args.get('path', '')
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        subprocess.Popen(['open', path])
    return f"Opening folder... <br><br><a href='/'>← Back to Home</a>"

@app.route('/open_file')
def open_file():
    path = request.args.get('path', '')
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        subprocess.Popen(['open', path])
    return f"Opening file... <br><br><a href='/'>← Back to Home</a>"

# --- HELPER FUNCTIONS ---
def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

if __name__ == '__main__':
    app.run(debug=True, port=5000)