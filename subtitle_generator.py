#!/usr/bin/env python3
"""
Enhanced Audio Subtitle Generator
Based on NeuralFalcon's Auto-Subtitle-Generator
Simplified for local use with auto language detection and English translation
"""

import os
import sys
import gc
import uuid
import re
import shutil
import argparse
from pathlib import Path
import time

try:
    import torch
    from faster_whisper import WhisperModel
    from tqdm.auto import tqdm
except ImportError as e:
    print("Error: Required packages not installed.")
    print("Install with: pip install faster-whisper torch tqdm")
    print(f"Missing: {e}")
    sys.exit(1)

# Configuration
SUBTITLE_FOLDER = "./generated_subtitles"
TEMP_FOLDER = "./temp_audio"

def clean_file_name(file_path):
    """Generates a clean, unique file name to avoid path issues."""
    dir_name = os.path.dirname(file_path)
    base_name, extension = os.path.splitext(os.path.basename(file_path))
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
    shutil.copy(uploaded_file, cleaned_path)
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
            formatted_lines = split_line_by_char_limit(text, max_chars_per_line)
            multiline_text = "\n".join(formatted_lines)
            srt_file.write(f"{index}\n{start} --> {end}\n{multiline_text}\n\n")

def generate_subtitles(audio_file, output_dir=None, model_size="base"):
    """
    Generate subtitles from audio file using faster-whisper
    Args:
        audio_file (str): Path to audio file
        output_dir (str): Output directory (default: same as input file)
        model_size (str): Whisper model size (tiny, base, small, medium, large-v1, large-v2, large-v3, large-v3-turbo)
    Returns:
        tuple: (original_srt_path, multiline_srt_path, transcript_text, detected_language)
    """
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    audio_path = Path(audio_file)
    if output_dir is None:
        output_dir = Path(SUBTITLE_FOLDER)
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = audio_path.stem[:30]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    print(f"Using device: {device}")
    print(f"Loading Whisper model '{model_size}'...")
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
    audio_file_path = get_audio_file(audio_file)
    print(f"Transcribing audio file: {audio_file}")
    print("Auto-detecting language and translating to English...")
    print("This may take a while depending on file size and model...")
    start_time = time.time()
    try:
        segments, info = model.transcribe(
            audio_file_path,
            word_timestamps=True,
            task="translate",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        segments = list(segments)
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
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    end_time = time.time()
    print(f"Transcription completed in {end_time - start_time:.2f} seconds")
    print(f"Detected language: {detected_language}")
    sentence_timestamps, word_timestamps, transcript_text = format_segments(segments)
    unique_id = uuid.uuid4().hex[:6]
    original_srt_path = output_dir / f"{base_name}_{detected_language}_{unique_id}_original.srt"
    multiline_srt_path = output_dir / f"{base_name}_{detected_language}_{unique_id}_multiline.srt"
    generate_srt_from_sentences(sentence_timestamps, str(original_srt_path))
    create_multiline_srt(sentence_timestamps, str(multiline_srt_path))
    print(f"\nSubtitles generated successfully!")
    print(f"🎯 Original Subtitles: {original_srt_path}")
    print(f"📝 Readable Subtitles: {multiline_srt_path}")
    print(f"🌍 Detected Language: {detected_language}")
    print(f"📄 Transcript Length: {len(transcript_text)} characters")
    return str(original_srt_path), str(multiline_srt_path), transcript_text, detected_language

def main():
    parser = argparse.ArgumentParser(description="Generate English subtitles from audio files using faster-whisper")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("-o", "--output", help="Output directory (default: ./generated_subtitles)")
    parser.add_argument("-m", "--model", default="base",
                       choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"],
                       help="Whisper model size (default: base)")
    args = parser.parse_args()
    try:
        generate_subtitles(args.audio_file, args.output, args.model)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    os.makedirs(SUBTITLE_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    if len(sys.argv) == 1:
        print("=== Enhanced Audio Subtitle Generator ===")
        print("Auto-detects language and translates to English\n")
        audio_file = input("Enter path to audio file: ").strip().strip('"\'')
        if not audio_file:
            print("No file specified. Exiting.")
            sys.exit(1)
        model_size = input("Enter model size (tiny/base/small/medium/large-v3-turbo) [base]: ").strip()
        if not model_size:
            model_size = "base"
        output_dir = input("Enter output directory (press Enter for ./generated_subtitles): ").strip()
        if not output_dir:
            output_dir = None
        try:
            generate_subtitles(audio_file, output_dir, model_size)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        main()
