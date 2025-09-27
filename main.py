from fastapi import FastAPI, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import os
import json
import uuid
from datetime import datetime
from typing import Optional, List
import logging
from pathlib import Path

# For PDFs
import fitz  # PyMuPDF

# For Audio
import speech_recognition as sr
from pydub import AudioSegment
from pydub.utils import make_chunks

# For MongoDB (optional)
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="File Processing API",
    description="API for extracting text from PDF and audio files",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_PDF_EXTENSIONS = {".pdf"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".mp4", ".m4a", ".flac", ".ogg"}
CHUNK_DURATION_MS = 60000  # 1 minute chunks for large audio files

# MongoDB setup
MONGO_URL = os.getenv("MONGO_URL", None)
mongo_client = None
mongo_collection = None

if MONGO_URL:
    try:
        mongo_client = MongoClient(MONGO_URL)
        # Test connection
        mongo_client.admin.command('ping')
        db = mongo_client["fastapi_db"]
        mongo_collection = db["uploads"]
        logger.info("✅ Connected to MongoDB")
    except Exception as e:
        logger.error(f"⚠️ MongoDB connection failed: {e}")
        mongo_collection = None

# Local storage setup
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

class ProcessingResult(BaseModel):
    id: str = Field(..., description="Unique identifier for the processed file")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Type of file processed")
    content: str = Field(..., description="Extracted text content")
    timestamp: str = Field(..., description="Processing timestamp")
    file_size: int = Field(..., description="File size in bytes")
    processing_duration: float = Field(..., description="Processing time in seconds")

class UploadResponse(BaseModel):
    status: str
    message: str
    result: ProcessingResult

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    mongodb_connected: bool

def validate_file_size(file: UploadFile) -> None:
    """Validate file size"""
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size allowed: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )

def get_file_extension(filename: str) -> str:
    """Get file extension safely"""
    return Path(filename).suffix.lower() if filename else ""

def generate_safe_filename(original_filename: str) -> tuple[str, str]:
    """Generate a safe filename with unique identifier"""
    file_id = str(uuid.uuid4())
    extension = get_file_extension(original_filename)
    safe_filename = f"{file_id}{extension}"
    return file_id, safe_filename

def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from PDF with error handling"""
    try:
        text_content = []
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                try:
                    page_text = page.get_text()
                    if page_text.strip():
                        text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                    text_content.append(f"--- Page {page_num + 1} ---\n[Error extracting text from this page]")
        
        result = "\n\n".join(text_content)
        return result.strip() if result.strip() else "No readable text found in PDF"
    
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_audio(file_path: Path) -> str:
    """Extract text from audio with chunking for large files"""
    recognizer = sr.Recognizer()
    
    try:
        # Load audio file
        audio = AudioSegment.from_file(str(file_path))
        
        # If audio is longer than chunk duration, process in chunks
        if len(audio) > CHUNK_DURATION_MS:
            chunks = make_chunks(audio, CHUNK_DURATION_MS)
            transcripts = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Export chunk to temporary wav file
                    chunk_path = OUTPUT_DIR / f"temp_chunk_{i}.wav"
                    chunk.export(chunk_path, format="wav")
                    
                    # Transcribe chunk
                    with sr.AudioFile(str(chunk_path)) as source:
                        audio_data = recognizer.record(source)
                        try:
                            chunk_text = recognizer.recognize_google(audio_data, language='en-US')
                            transcripts.append(f"[Chunk {i+1}] {chunk_text}")
                        except sr.UnknownValueError:
                            transcripts.append(f"[Chunk {i+1}] [Inaudible]")
                        except sr.RequestError as e:
                            transcripts.append(f"[Chunk {i+1}] [API Error: {e}]")
                    
                    # Clean up temporary file
                    chunk_path.unlink(missing_ok=True)
                    
                except Exception as e:
                    logger.warning(f"Error processing audio chunk {i}: {e}")
                    transcripts.append(f"[Chunk {i+1}] [Processing Error]")
            
            return "\n\n".join(transcripts) if transcripts else "No speech detected in audio"
        
        else:
            # Process entire file for shorter audio
            with sr.AudioFile(str(file_path)) as source:
                audio_data = recognizer.record(source)
                try:
                    return recognizer.recognize_google(audio_data, language='en-US')
                except sr.UnknownValueError:
                    return "No speech could be recognized in the audio"
                except sr.RequestError as e:
                    return f"Speech recognition API error: {e}"
    
    except Exception as e:
        logger.error(f"Audio extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text from audio: {str(e)}")

def save_result(result: ProcessingResult) -> str:
    """Save processing result to MongoDB or local storage"""
    result_dict = result.dict()
    
    if mongo_collection:
        try:
            mongo_collection.insert_one(result_dict)
            return "stored_in_mongodb"
        except Exception as e:
            logger.error(f"MongoDB storage error: {e}")
            # Fallback to local storage
    
    # Local storage fallback
    try:
        json_path = OUTPUT_DIR / f"{result.id}_result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        return "stored_locally"
    except Exception as e:
        logger.error(f"Local storage error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save processing result")

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        mongodb_connected=mongo_collection is not None
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "File Processing API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload/",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/upload/", response_model=UploadResponse)
async def upload_file(file: UploadFile):
    """
    Upload and process PDF or audio files to extract text content.
    
    Supported formats:
    - PDF: .pdf
    - Audio: .mp3, .wav, .mp4, .m4a, .flac, .ogg
    """
    start_time = datetime.now()
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    validate_file_size(file)
    
    file_ext = get_file_extension(file.filename)
    if file_ext not in ALLOWED_PDF_EXTENSIONS and file_ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported types: {list(ALLOWED_PDF_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS)}"
        )
    
    # Generate safe filename and read file content
    file_id, safe_filename = generate_safe_filename(file.filename)
    file_path = OUTPUT_DIR / safe_filename
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            file_size = len(content)
        
        # Extract text based on file type
        if file_ext in ALLOWED_PDF_EXTENSIONS:
            extracted_content = extract_text_from_pdf(file_path)
            file_type = "pdf"
        elif file_ext in ALLOWED_AUDIO_EXTENSIONS:
            extracted_content = extract_text_from_audio(file_path)
            file_type = "audio"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Calculate processing duration
        processing_duration = (datetime.now() - start_time).total_seconds()
        
        # Create result object
        result = ProcessingResult(
            id=file_id,
            filename=file.filename,
            file_type=file_type,
            content=extracted_content,
            timestamp=start_time.isoformat(),
            file_size=file_size,
            processing_duration=processing_duration
        )
        
        # Save result
        storage_status = save_result(result)
        
        return UploadResponse(
            status="success",
            message=f"File processed successfully and {storage_status}",
            result=result
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Clean up uploaded file
        try:
            file_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to clean up file {file_path}: {e}")

@app.get("/uploads/", response_model=List[ProcessingResult])
async def get_uploads(limit: int = 10):
    """Get recent uploads from storage"""
    if mongo_collection:
        try:
            cursor = mongo_collection.find().sort("timestamp", -1).limit(limit)
            results = [ProcessingResult(**doc) for doc in cursor]
            return results
        except Exception as e:
            logger.error(f"Error fetching from MongoDB: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch uploads from database")
    else:
        # Local storage fallback
        try:
            json_files = list(OUTPUT_DIR.glob("*_result.json"))
            json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            results = []
            for json_file in json_files[:limit]:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        results.append(ProcessingResult(**data))
                except Exception as e:
                    logger.warning(f"Error reading {json_file}: {e}")
            
            return results
        except Exception as e:
            logger.error(f"Error fetching local uploads: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch uploads from local storage")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Assuming this file is named main.py
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )