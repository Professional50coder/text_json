import os
import sys
import gc
import uuid
import re
import shutil
from pathlib import Path
import time


# --- Constants ---
GOOGLE_API_KEY = 'AIzaSyAF-7OArGZV9evzIebZg2vIO8I_UNZeZZs'
DEFAULT_AUDIO_FILE = 'Audio_Gujrati_sample1.m4a'
DEFAULT_OUTPUT_DIR = './generated_subtitles'
DEFAULT_MODEL = 'large-v3-turbo'
TEMP_FOLDER = './temp_audio_files'
SUBTITLE_FOLDER = './generated_subtitles'

# --- Dependency Checks ---
try:
    import torch
    from faster_whisper import WhisperModel
    from tqdm.auto import tqdm
except ImportError as e:
    print(f"Error: Required packages not installed.")
    print("Install with: pip install faster-whisper torch tqdm")
    print(f"Missing: {e}")
    sys.exit(1)

try:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
except ImportError:
    print("Warning: google.generativeai not installed. Some features may not work.")

# --- Utility Functions ---

def clean_file_name(file_path):
    """Generates a clean, unique file name to avoid path issues."""
    dir_name = os.path.dirname(file_path)
    base_name, extension = os.path.splitext(os.path.basename(file_path))

    # Clean the base name
    cleaned_base = re.sub(r'[^a-zA-Z\d]+', '_', base_name)
    cleaned_base = re.sub(r'_+', '_', cleaned_base).strip('_')
    random_uuid = uuid.uuid4().hex[:6]

    return os.path.join(dir_name, f"{cleaned_base}_{random_uuid}{extension}")

def convert_time_to_srt_format(seconds):
    """Converts seconds to the standard SRT time format (HH:MM:SS,ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = round((seconds - int(seconds)) * 1000)

    if milliseconds == 1000:
        milliseconds = 0
        secs += 1
        if secs == 60:
            secs, minutes = 0, minutes + 1
            if minutes == 60:
                minutes, hours = 0, hours + 1

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def split_line_by_char_limit(text, max_chars_per_line=38):
    """Splits a string into multiple lines based on a character limit."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if not current_line:
            current_line = word
        elif len(current_line + " " + word) <= max_chars_per_line:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines

def format_segments(segments):
    """Formats the raw segments from Whisper into structured lists."""
    sentence_timestamps = []
    word_timestamps = []
    speech_to_text = ""

    for i, segment in enumerate(segments):
        text = segment.text.strip()
        sentence_timestamps.append({
            "id": i,
            "text": text,
            "start": segment.start,
            "end": segment.end
        })
        speech_to_text += text + " "

        # Handle word-level timestamps if available
        if hasattr(segment, 'words') and segment.words:
            for word in segment.words:
                word_timestamps.append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end
                })

    return sentence_timestamps, word_timestamps, speech_to_text.strip()

def get_audio_file(uploaded_file):
    """Copies the uploaded media file to a temporary location for processing."""
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    temp_path = os.path.join(TEMP_FOLDER, os.path.basename(uploaded_file))
    cleaned_path = clean_file_name(temp_path)
    print(f"[INFO] Copying audio file to temp location: {cleaned_path}")
    start = time.time()
    try:
        shutil.copy(uploaded_file, cleaned_path)
    except Exception as e:
        print(f"Error copying audio file: {e}")
        raise
    print(f"[INFO] Audio file copied in {time.time() - start:.2f} seconds.")
    return cleaned_path

def generate_srt_from_sentences(sentence_timestamps, srt_path):
    """Generates a standard SRT file from sentence-level timestamps."""
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for index, sentence in enumerate(sentence_timestamps, start=1):
            start = convert_time_to_srt_format(sentence['start'])
            end = convert_time_to_srt_format(sentence['end'])
            srt_file.write(f"{index}\n{start} --> {end}\n{sentence['text']}\n\n")

def create_multiline_srt(sentence_timestamps, srt_path, max_chars_per_line=38):
    """Creates readable multi-line SRT file content"""
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        for index, sentence in enumerate(sentence_timestamps, start=1):
            start = convert_time_to_srt_format(sentence['start'])
            end = convert_time_to_srt_format(sentence['end'])
            text = sentence['text'].strip()

            # Split long text into multiple lines
            formatted_lines = split_line_by_char_limit(text, max_chars_per_line)
            multiline_text = "\n".join(formatted_lines)

            srt_file.write(f"{index}\n{start} --> {end}\n{multiline_text}\n\n")

def generate_subtitles(audio_file, output_dir=None, model_size="base"):
    """
    Generate subtitles from audio file using faster-whisper

    Args:
        audio_file (str): Path to audio file
        output_dir (str): Output directory (default: same as input file)
        model_size (str): Whisper model size (tiny, base, small, medium, large-v1, large-v2, large-v3)

    Returns:
        tuple: (original_srt_path, multiline_srt_path, transcript_text, detected_language)
    """

    # Validate input file
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    audio_path = Path(audio_file)
    if output_dir is None:
        output_dir = Path(SUBTITLE_FOLDER)
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = audio_path.stem[:30]

    # Configure device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    print(f"Using device: {device}")
    print(f"Loading Whisper model '{model_size}'...")

    model = None
    model_load_start = time.time()
    try:
        if model_size == "large-v3-turbo":
            print("Attempting to use faster-whisper large-v3-turbo...")
            model = WhisperModel("deepdml/faster-whisper-large-v3-turbo-ct2", device=device, compute_type=compute_type)
        else:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to base model...")
        try:
            model = WhisperModel("base", device=device, compute_type=compute_type)
            model_size = "base"
        except Exception as e2:
            print(f"Error loading fallback model: {e2}")
            return None, None, None, None
    print(f"[INFO] Model loaded in {time.time() - model_load_start:.2f} seconds.")

    # Process audio file
    temp_audio_file_path = get_audio_file(audio_file)

    print(f"Transcribing audio file: {audio_file}")
    print("Auto-detecting language and translating to English...")
    print("This may take a while depending on file size and model...")

    transcribe_start = time.time()
    try:
        # tqdm progress bar for generator if possible
        segments_gen, info = model.transcribe(
            temp_audio_file_path,
            word_timestamps=True,
            task="translate",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        print("[INFO] Collecting segments...")
        segments = list(tqdm(segments_gen, desc="Transcribing", unit="segment"))
        detected_language = info.language
    except Exception as e:
        print(f"Error during transcription: {e}")
        if "The system cannot find the file specified" in str(e):
            print("\n💡 SOLUTION:")
            print("This error usually means FFmpeg is not installed.")
            print("Install FFmpeg using one of these methods:")
            print("1. choco install ffmpeg (if you have Chocolatey)")
            print("2. Download from https://ffmpeg.org/download.html")
            print("3. pip install ffmpeg-python")
            print("4. Or convert your audio file to .wav format first")
        return None, None, None, None
    finally:
        # Cleanup
        if temp_audio_file_path and os.path.exists(temp_audio_file_path):
            try:
                os.remove(temp_audio_file_path)
            except Exception:
                pass
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    transcribe_end = time.time()
    print(f"[INFO] Transcription completed in {transcribe_end - transcribe_start:.2f} seconds")
    print(f"Detected language: {detected_language}")

    # Format segments
    sentence_timestamps, word_timestamps, transcript_text = format_segments(segments)

    # Generate file paths
    unique_id = uuid.uuid4().hex[:6]
    original_srt_path = output_dir / f"{base_name}_{detected_language}_{unique_id}_original.srt"
    multiline_srt_path = output_dir / f"{base_name}_{detected_language}_{unique_id}_multiline.srt"

    # Generate SRT files
    srt_write_start = time.time()
    print(f"[INFO] Writing original SRT file: {original_srt_path}")
    generate_srt_from_sentences(sentence_timestamps, str(original_srt_path))
    print(f"[INFO] Writing multiline SRT file: {multiline_srt_path}")
    create_multiline_srt(sentence_timestamps, str(multiline_srt_path))
    print(f"[INFO] SRT files written in {time.time() - srt_write_start:.2f} seconds.")

    print(f"\nSubtitles generated successfully!")
    print(f"🎯 Original Subtitles: {original_srt_path}")
    print(f"📝 Readable Subtitles: {multiline_srt_path}")
    print(f"🌍 Detected Language: {detected_language}")
    print(f"📄 Transcript Length: {len(transcript_text)} characters")

    return str(original_srt_path), str(multiline_srt_path), transcript_text, detected_language

# --- Main Execution ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audio to Subtitle Generator (faster-whisper)")
    parser.add_argument("--audio", type=str, default=DEFAULT_AUDIO_FILE, help="Path to audio file")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Whisper model size")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    original_srt, multiline_srt, transcript, language = generate_subtitles(
        args.audio, args.output, args.model
    )

    if original_srt:
        print(f"\nOriginal SRT file: {original_srt}")
        print(f"Multiline SRT file: {multiline_srt}")
        print(f"Detected Language: {language}")
        print("\nTranscript:")
        print(transcript)