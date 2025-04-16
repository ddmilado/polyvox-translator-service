import os
import uuid
import json
import time
import signal
import sys
import logging
import threading
import shutil
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client

# Import Crew AI components.
# Adjust these import paths as needed for your environment.
from crewai import LLM, Agent, Task, Crew, Process

# ========= CONFIGURATION =========

# Environment variables must be defined (consider using a .env file or export them)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_API_KEY:
    sys.exit("Please set SUPABASE_URL, SUPABASE_KEY, and OPENAI_API_KEY environment variables.")

# Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
SUPABASE_BUCKET = "translations"  # Change if needed

# Translation settings
CHUNK_SIZE = 4000  # Characters per chunk (adjust as necessary)
MAX_TOTAL_LENGTH = 100000  # Maximum allowed characters in the source text

# Temporary directories for job files
TEMP_DIR = os.path.join(tempfile.gettempdir(), "translation_service")
RESULTS_DIR = os.path.join(TEMP_DIR, "results")
for directory in [TEMP_DIR, RESULTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# ========= LOGGING =========

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ========= GLOBAL JOB STORAGE =========

jobs = {}  # job_id -> job_data dictionary
job_lock = threading.Lock()

# ========= UTILITY FUNCTIONS =========

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """
    Split text into chunks of at most chunk_size characters.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def save_result_to_file(job_id, result_data):
    """
    Save the translation result (a dictionary) as a JSON file.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"{job_id}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved job result to {result_path}")
    return result_path

def upload_to_supabase(file_path, remote_filename):
    """
    Read a local file and upload it to Supabase Storage.
    """
    with open(file_path, "rb") as f:
        file_data = f.read()
    logger.info(f"Uploading {remote_filename} ({len(file_data)} bytes) to Supabase bucket {SUPABASE_BUCKET}...")
    response = supabase.storage.from_(SUPABASE_BUCKET).upload(remote_filename, file_data)
    logger.info(f"Upload response: {response}")
    return response

def update_job_data(job_id, data):
    with job_lock:
        if job_id in jobs:
            jobs[job_id].update(data)
        else:
            jobs[job_id] = data

def get_job_data(job_id):
    with job_lock:
        return jobs.get(job_id)

def clean_up_job(job_id):
    """
    Remove job from memory and schedule deletion of its result file.
    """
    with job_lock:
        if job_id in jobs:
            del jobs[job_id]
    result_file = os.path.join(RESULTS_DIR, f"{job_id}.json")
    if os.path.exists(result_file):
        def delete_file():
            try:
                os.remove(result_file)
                logger.info(f"Cleaned up result file for job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to delete result file for job {job_id}: {e}")
        t = threading.Timer(3600, delete_file)  # Delete after 1 hour
        t.daemon = True
        t.start()

# ========= CREW AI TRANSLATION FUNCTION =========

def translate_chunk_with_crewai(chunk_text, source_language, target_language, api_key):
    """
    Use Crew AI to translate a text chunk.
    """
    # Ensure the API key is set for the Crew AI components.
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        # Initialize the LLM with a timeout of 120 seconds.
        llm = LLM(model="gpt-3.5-turbo", timeout=120)
        
        # Create a Translator agent.
        translator = Agent(
            role="Translator",
            goal=f"Translate the given text from {source_language} to {target_language}.",
            backstory="Expert translator",
            verbose=False,
            allow_delegation=False,
            llm=llm
        )
        
        # Create an Editor agent to refine the translation.
        editor = Agent(
            role="Editor",
            goal=f"Refine the translation to {target_language} ensuring clarity and style.",
            backstory="Skilled language editor",
            verbose=False,
            allow_delegation=False,
            llm=llm
        )
        
        # Define the translation task.
        translate_task = Task(
            description=f"Translate from {source_language} to {target_language}:\n\n{chunk_text}",
            agent=translator,
            expected_output="Translated text"
        )
        
        # Define the edit task.
        edit_task = Task(
            description=f"Refine and polish the translation to {target_language}.",
            agent=editor,
            expected_output="Polished translation"
        )
        
        # Create and execute a Crew with both agents, running tasks sequentially.
        translation_crew = Crew(
            agents=[translator, editor],
            tasks=[translate_task, edit_task],
            verbose=False,
            process=Process.sequential
        )
        
        result = translation_crew.kickoff()
        # If the result has a 'raw' attribute, return that.
        if hasattr(result, "raw"):
            return result.raw
        return str(result)
    
    except Exception as e:
        logger.error(f"Crew AI error: {e}")
        raise Exception(f"Crew AI translation error: {e}")

# ========= JOB PROCESSING =========

def process_translation_job(job_id, source_text, source_language, target_language, api_key):
    """
    Split the source text into chunks, translate each chunk using Crew AI,
    save the final translation to a JSON file, and upload it to Supabase.
    """
    try:
        update_job_data(job_id, {"status": "processing", "chunks_completed": 0, "total_chunks": 0})
        
        # Check if the source text is too large.
        if len(source_text) > MAX_TOTAL_LENGTH:
            error_msg = f"Source text exceeds maximum allowed length of {MAX_TOTAL_LENGTH} characters."
            update_job_data(job_id, {"status": "failed", "error": error_msg})
            return
        
        # Split the source text into chunks.
        chunks = chunk_text(source_text, CHUNK_SIZE)
        update_job_data(job_id, {"total_chunks": len(chunks)})
        logger.info(f"Job {job_id}: Split source text into {len(chunks)} chunks.")
        
        translated_chunks = []
        for idx, chunk in enumerate(chunks):
            logger.info(f"Job {job_id}: Translating chunk {idx+1}/{len(chunks)}...")
            try:
                translated = translate_chunk_with_crewai(chunk, source_language, target_language, api_key)
                translated_chunks.append(translated)
                update_job_data(job_id, {"chunks_completed": idx+1})
            except Exception as e:
                translated_chunks.append(f"[ERROR in chunk {idx+1}: {e}]")
                logger.error(f"Job {job_id}: Error in chunk {idx+1}: {e}")
        
        final_translation = "\n\n".join(translated_chunks)
        job_result = {
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "completed",
            "success": True,
            "translation": final_translation,
            "metadata": {
                "total_chunks": len(chunks),
                "source_language": source_language,
                "target_language": target_language
            }
        }
        
        # Save job result locally.
        result_path = save_result_to_file(job_id, job_result)
        update_job_data(job_id, {"status": "completed", "result_file": result_path})
        
        # Upload the result file to Supabase Storage.
        try:
            upload_to_supabase(result_path, f"{job_id}.json")
        except Exception as upload_error:
            logger.error(f"Job {job_id}: Supabase upload error: {upload_error}")
            job_result["status"] = "upload_failed"
            update_job_data(job_id, {"status": "upload_failed", "error": str(upload_error)})
        
    except Exception as e:
        logger.exception(f"Job {job_id}: Exception during processing.")
        update_job_data(job_id, {"status": "failed", "error": str(e)})
    finally:
        # Schedule cleanup after 1 hour.
        threading.Timer(3600, lambda: clean_up_job(job_id)).start()

# ========= FLASK APP =========

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Translation service is running.",
        "version": "1.0"
    })

@app.route("/translate", methods=["POST"])
def translate():
    """
    Expected JSON payload:
    {
      "sourceText": "Text to translate...",
      "sourceLanguage": "Norwegian",
      "targetLanguage": "English",
      "openaiApiKey": "your-openai-api-key"
    }
    """
    try:
        data = request.get_json()
        required_fields = ["sourceText", "sourceLanguage", "targetLanguage", "openaiApiKey"]
        if not all(data.get(field) for field in required_fields):
            return jsonify({"success": False, "error": "Missing required fields."}), 400
        
        source_text = data.get("sourceText")
        source_language = data.get("sourceLanguage")
        target_language = data.get("targetLanguage")
        openai_api_key = data.get("openaiApiKey")
        
        job_id = str(uuid.uuid4())
        # Store initial job data.
        update_job_data(job_id, {
            "job_id": job_id,
            "status": "created",
            "created_at": datetime.utcnow().isoformat(),
            "source_language": source_language,
            "target_language": target_language,
            "chunks_completed": 0,
            "total_chunks": 0
        })
        
        # Start background processing.
        thread = threading.Thread(
            target=process_translation_job,
            args=(job_id, source_text, source_language, target_language, openai_api_key)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "created"
        })
    
    except Exception as e:
        logger.exception("Error starting translation job.")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/job/<job_id>", methods=["GET"])
def job_status(job_id):
    """
    Return the current status and job data.
    """
    try:
        result_file = os.path.join(RESULTS_DIR, f"{job_id}.json")
        if os.path.exists(result_file):
            with open(result_file, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            return jsonify(result_data)
        job_data = get_job_data(job_id)
        if job_data:
            return jsonify(job_data)
        return jsonify({"success": False, "error": "Job not found"}), 404
    except Exception as e:
        logger.exception(f"Error retrieving status for job {job_id}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    """
    Mark a job as cancelled.
    """
    try:
        job_data = get_job_data(job_id)
        if not job_data:
            return jsonify({"success": False, "error": "Job not found"}), 404
        
        update_job_data(job_id, {"status": "cancelled"})
        result_data = {
            "success": False,
            "status": "cancelled",
            "error": "Job cancelled by user",
            "timestamp": datetime.utcnow().isoformat()
        }
        save_result_to_file(job_id, result_data)
        return jsonify({"success": True, "job_id": job_id, "status": "cancelled"})
    
    except Exception as e:
        logger.exception(f"Error cancelling job {job_id}")
        return jsonify({"success": False, "error": str(e)}), 500

# ========= SHUTDOWN HANDLING =========

def cleanup_on_exit(*args, **kwargs):
    logger.info("Cleaning up before shutdown...")
    with job_lock:
        for job_id, job_data in jobs.items():
            if job_data.get("status") in ["created", "processing"]:
                job_data["status"] = "interrupted"
                result_data = {
                    "success": False,
                    "status": "interrupted",
                    "error": "Service shutdown before job completion",
                    "timestamp": datetime.utcnow().isoformat()
                }
                save_result_to_file(job_id, result_data)
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_on_exit)
signal.signal(signal.SIGTERM, cleanup_on_exit)

# ========= MAINTENANCE THREAD =========

def maintenance_thread():
    while True:
        try:
            now = time.time()
            for filename in os.listdir(RESULTS_DIR):
                filepath = os.path.join(RESULTS_DIR, filename)
                if os.path.isfile(filepath) and now - os.path.getmtime(filepath) > 86400:
                    try:
                        os.remove(filepath)
                        logger.info(f"Removed old result file: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old file {filename}: {e}")
        except Exception as e:
            logger.error(f"Maintenance thread error: {e}")
        time.sleep(3600)

maintenance = threading.Thread(target=maintenance_thread)
maintenance.daemon = True
maintenance.start()

# ========= MAIN =========

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # Clean temporary directories on startup
    try:
        for item in os.listdir(TEMP_DIR):
            item_path = os.path.join(TEMP_DIR, item)
            if os.path.isdir(item_path) and item != "results":
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                os.remove(item_path)
    except Exception as e:
        logger.warning(f"Error cleaning temporary directory: {e}")
    
    # Run the Flask server
    app.run(host="0.0.0.0", port=port, threaded=True)
