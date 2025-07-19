from flask import Flask, render_template, request, jsonify, send_file
import requests
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch
import os
import uuid
from nltk.tokenize import sent_tokenize
import nltk
import warnings
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import PyPDF2  # Added for PDF support


# Ensure NLTK resources are downloaded
nltk.download('punkt')
app = Flask(__name__, static_folder='static')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-testing')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Add FFmpeg executable path
#os.environ[
#    "PATH"] = r"C:\Users\Ritika\Downloads\QuickNotes\QuickNotes\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin;" + \
#              os.environ["PATH"]

# Model configuration
MODEL_CONFIG = {
    'summarization': 'facebook/bart-large-cnn',  # Will be replaced with a smaller model
    'qa_generation': 'google/flan-t5-base'  # Will be replaced with a smaller model
}

# Use smaller models for Render compatibility
USE_SMALL_MODELS = True  # Set to False for local development with larger models

if USE_SMALL_MODELS:
    MODEL_CONFIG.update({
        'summarization': 'sshleifer/distilbart-cnn-12-6',  # Much smaller model for summarization
        'qa_generation': 'mrm8488/t5-base-finetuned-question-generation-ap'  # Smaller QG model
    })

# Initialize models as None for lazy loading
summarizer = None
qa_generator = None

def load_models():
    """Lazily load models only when needed"""
    global summarizer, qa_generator
    
    if summarizer is None:
        print("Loading summarization model...")
        summarizer = pipeline(
            "summarization",
            model=MODEL_CONFIG['summarization'],
            tokenizer=MODEL_CONFIG['summarization'],
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    
    if qa_generator is None:
        print("Loading QA generation model...")
        qa_generator = pipeline(
            "text2text-generation",
            model=MODEL_CONFIG['qa_generation'],
            tokenizer=MODEL_CONFIG['qa_generation'],
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    
    return summarizer, qa_generator


def extract_video_id(url):
    """Extract YouTube video ID from various URL formats."""
    try:
        parsed_url = urlparse(url)
        if "youtu.be" in parsed_url.netloc:  # Shortened URL
            return parsed_url.path.lstrip('/')
        elif "youtube.com" in parsed_url.netloc:  # Full URL
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        return None
    except Exception as e:
        print(f"Error extracting video ID: {e}")
        return None


def get_youtube_transcript(url):
    """Get transcript from YouTube video using youtube_transcript_api."""
    try:
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Could not extract video ID from URL")

        print(f"Getting transcript for video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

        # Combine all text parts into a single transcript
        full_transcript = " ".join([item['text'] for item in transcript_list])
        return full_transcript
    except Exception as e:
        print(f"YouTube transcript error: {e}")
        return None


def download_youtube_audio(url):
    """Download audio from YouTube videos - kept for audio processing if needed."""
    try:
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("Unable to extract YouTube video ID.")

        # Generate a unique filename
        filename = f"{uuid.uuid4()}.mp4"
        output_path = "temp"

        # Create the directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # For now, just return the ID as we'll be using transcript API instead
        return video_id
    except Exception as e:
        print(f"YouTube download error: {e}")
        return None


def extract_text_from_pdf(filepath):
    """Extracts all text from a PDF file."""
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return None


def transcribe_audio(filepath):
    """
    Transcribe an audio/video file using AssemblyAI API.
    """
    print("Transcribing using AssemblyAI...")
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("AssemblyAI API key not set.")
        return None
    headers = {"authorization": api_key}
    # 1. Upload file to AssemblyAI
    with open(filepath, "rb") as f:
        response = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers=headers,
            data=f
        )
    if response.status_code != 200:
        print("Upload failed:", response.text)
        return None
    upload_url = response.json()["upload_url"]
    # 2. Request transcription
    transcript_response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=headers,
        json={"audio_url": upload_url}
    )
    if transcript_response.status_code != 200:
        print("Transcription request failed:", transcript_response.text)
        return None
    transcript_id = transcript_response.json()["id"]
    # 3. Poll for completion
    polling_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    import time
    for _ in range(60):  # Wait up to ~3 minutes
        poll_res = requests.get(polling_url, headers=headers)
        status = poll_res.json()["status"]
        if status == "completed":
            return poll_res.json()["text"]
        elif status == "failed":
            print("Transcription failed:", poll_res.text)
            return None
        time.sleep(3)
    print("Transcription timed out.")
    return None



def preprocess_text(text):
    print("Preprocessing transcript...")
    try:
        sentences = sent_tokenize(text)
        cleaned_sentences = [sentence.strip() for sentence in sentences if len(sentence.split()) > 3]
        preprocessed_text = " ".join(cleaned_sentences)
        print("Preprocessed Text:", preprocessed_text)
        return preprocessed_text
    except Exception as e:
        print("Error during preprocessing:", e)
        return text


def summarize_text(text):
    print("Summarizing...")
    try:
        # Ensure models are loaded
        summarizer, _ = load_models()
        
        preprocessed_text = preprocess_text(text)
        chunks = [preprocessed_text[i:i + 1000] for i in range(0, len(preprocessed_text), 1000)]
        summaries = []
        for chunk in chunks:
            input_length = len(chunk.split())  # Count words
            max_len = max(50, int(input_length * 0.8))  # Adjust max_length to 80% of input length
            summary = summarizer(chunk, max_length=max_len, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        summarized_text = " ".join(summaries)
        return summarized_text
    except Exception as e:
        print("Error during summarization:", e)
        return "Error during summarization."


import traceback

def generate_questions_from_summary(summary):
    print("Generating questions from summary...")
    print("Summary input:", summary)
    try:
        # Ensure models are loaded
        _, qa_generator = load_models()
        
        sentences = sent_tokenize(summary)
        print("Tokenized sentences:", sentences)
        if not sentences:
            print("No sentences found in summary.")
            return ["No questions generated: summary was empty."]
        questions = []
        for sentence in sentences[:5]:  # Limit to first 5 sentences to reduce load
            try:
                qa = qa_generator(
                    f"Generate a question based on: {sentence}",
                    max_length=64,
                    num_return_sequences=1,
                    do_sample=True
                )[0]['generated_text']
                questions.append(qa)
            except Exception as e:
                print(f"Error generating question for sentence: {e}")
                continue
                
        if not questions:  # If all generations failed, return a helpful message
            return ["Could not generate questions. The content might be too short or unclear."]
            
        return questions
    except Exception as e:
        print("Error generating questions:", e)
        traceback.print_exc()
        return [f"Error generating questions: {str(e)}"]


def save_to_file(content, filename):
    """Save the given content to a text file."""
    filepath = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)
    return filepath


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/remove-file', methods=['POST'])
def remove_file():
    try:
        form_data = request.get_json()
        if not form_data or 'filename' not in form_data:
            return jsonify({"error": "No filename provided"}), 400

        filename = form_data['filename']
        filepath = os.path.join("temp", filename)

        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"success": True})
        else:
            return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/process', methods=['POST'])
def process():
    import time
    try:
        start_time = time.time()
        file = request.files.get('file')
        youtube_url = request.form.get('youtube_url')
        transcript = None
        step_times = {}

        if not file and not youtube_url:
            return jsonify({"error": "No file or URL provided"}), 400

        # Handle file upload
        if file:
            filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = os.path.join("temp", filename)
            os.makedirs("temp", exist_ok=True)
            file.save(filepath)
            # Check if PDF
            if filename.lower().endswith('.pdf'):
                t0 = time.time()
                transcript = extract_text_from_pdf(filepath)
                step_times['pdf_extract'] = time.time() - t0
                if not transcript:
                    return jsonify({"error": "Failed to extract text from PDF"}), 400
            else:
                t0 = time.time()
                transcript = transcribe_audio(filepath)
                step_times['audio_transcribe'] = time.time() - t0

        # Handle YouTube URL
        elif youtube_url:
            t0 = time.time()
            transcript = get_youtube_transcript(youtube_url)
            step_times['youtube_transcript'] = time.time() - t0
            if not transcript:
                return jsonify({"error": "Failed to get YouTube transcript"}), 400

        if not transcript:
            return jsonify({"error": "Transcription failed"}), 500

        t0 = time.time()
        summary = summarize_text(transcript)
        step_times['summarize'] = time.time() - t0
        if not summary:
            return jsonify({"error": "Summarization failed"}), 500

        t0 = time.time()
        questions = generate_questions_from_summary(summary)
        step_times['question_gen'] = time.time() - t0

        # Save files
        transcript_file = save_to_file(transcript, "transcript.txt")
        summary_file = save_to_file(summary, "summary.txt")
        questions_file = save_to_file("\n".join(questions), "questions.txt")

        total_time = time.time() - start_time
        print("--- Processing Timings (seconds) ---")
        for k, v in step_times.items():
            print(f"{k}: {v:.2f}s")
        print(f"Total: {total_time:.2f}s")
        print("-----------------------------------")

        # Return file paths and the actual content
        return jsonify({
            "transcript_content": transcript,
            "summary_content": summary,
            "qa_content": questions,
            "transcript_file": f"/download/{os.path.basename(transcript_file)}",
            "summary_file": f"/download/{os.path.basename(summary_file)}",
            "qa_file": f"/download/{os.path.basename(questions_file)}"
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    filepath = os.path.join("temp", filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({"error": "File not found"}), 404


if __name__ == '__main__':
    # Get port from environment variable or default to 10000
    port = int(os.environ.get('PORT', 10000))
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)
