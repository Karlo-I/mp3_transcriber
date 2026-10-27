import os
import uuid
import subprocess
import whisper
import torch
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from urllib.parse import quote

app = Flask(__name__)

# Create a temporary folder to hold uploaded files during processing
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
    # 1. Handle the uploaded file
    if 'audio_file' not in request.files:
        return "Error: No file part in the request."
    
    file = request.files['audio_file']
    if file.filename == '':
        return "Error: No file selected."

    # 2. Handle the save directory
    save_dir = request.form.get('save_path', '').strip()
    
    # Default to Desktop if left blank
    if not save_dir:
        save_dir = os.path.expanduser("~/Desktop")
    else:
        save_dir = os.path.expanduser(save_dir)

    if not os.path.isdir(save_dir):
        return f"Error: The folder '{save_dir}' does not exist. Please check the path."

    # 3. Save the uploaded file temporarily
    temp_filename = f"temp_{uuid.uuid4()}_{secure_filename(file.filename)}"
    temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
    file.save(temp_filepath)

    try:
        # 4. Transcribe the audio
        print(f"Transcribing {file.filename}... (Check terminal for progress)")
        result = model.transcribe(temp_filepath, verbose=True) 

        # 5. Save the text file to the user's chosen directory with timestamps
        original_name = os.path.splitext(secure_filename(file.filename))[0]
        output_filename = f"{original_name}_transcript.txt"
        output_filepath = os.path.join(save_dir, output_filename)

        with open(output_filepath, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                start_time = format_timestamp(segment["start"])
                text = segment["text"].strip()
                f.write(f"[{start_time}] {text}\n\n")

        # 6. Create secure links to our new backend routes
        # We use urllib.parse.quote to safely handle spaces in file paths
        folder_link = f"/open_folder?path={quote(save_dir)}"
        file_link = f"/open_file?path={quote(output_filepath)}"

        # 7. Return the success page with clickable buttons
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
                <a href="{file_link}" class="btn btn-secondary"> Open Text File</a>
            </div>

            <br><br>
            <a href="/">← Transcribe another file</a>
        </body>
        </html>
        """

    except Exception as e:
        return f"Error during transcription: {str(e)}"
    
    finally:
        # 8. Clean up the temporary audio file to save space
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

# --- NEW ROUTES TO OPEN FILES/FOLDERS USING MACOS NATIVE COMMANDS ---

@app.route('/open_folder')
def open_folder():
    path = request.args.get('path', '')
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        # 'open' is the macOS command to open Finder
        subprocess.Popen(['open', path])
    return f"Opening folder... <br><br><a href='/'>← Back to Home</a>"

@app.route('/open_file')
def open_file():
    path = request.args.get('path', '')
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        # 'open' will open the file in your default text editor (TextEdit)
        subprocess.Popen(['open', path])
    return f"Opening file... <br><br><a href='/'>← Back to Home</a>"

# --- HELPER FUNCTIONS ---

def format_timestamp(seconds):
    """Helper to make seconds look like 00:00:00"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

if __name__ == '__main__':
    app.run(debug=True, port=5000)