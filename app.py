from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import time
from datetime import datetime
import tiktoken
import logging
import tempfile
import shutil
import threading
import json
import signal
import sys

# Import your Crew AI components (adjust the import paths as needed)
from crew_ai import LLM, Agent, Task, Crew, Process

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_CHUNKS = 100
MAX_TOKENS_PER_CHUNK = 800  # Further reduced chunk size
MAX_TOTAL_TOKENS = 100000
TOKEN_COST_PER_1K = 0.01
PROCESS_TIMEOUT = 90  # Process timeout in seconds
MAX_RETRIES = 3  # Allow retrying failed chunks
TEMP_DIR = os.path.join(tempfile.gettempdir(), "translation_service")
RESULTS_DIR = os.path.join(TEMP_DIR, "results")
MAX_CONCURRENT_PROCESSES = 2  # Limit concurrent translation processes

# Create temp directories if they don't exist
for directory in [TEMP_DIR, RESULTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Global job storage
jobs = {}
job_lock = threading.Lock()

# Function to estimate token count
def estimate_tokens(text, model="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error estimating tokens: {e}")
        # Fallback to character-based estimation
        return len(text) // 4

# Function to estimate translation cost
def estimate_cost(total_tokens):
    return (total_tokens / 1000) * TOKEN_COST_PER_1K

# Function to split text into smaller chunks
def split_text(text, max_tokens=MAX_TOKENS_PER_CHUNK):
    if not text:
        return []
        
    # Simple splitting by paragraphs first
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = []
    current_chunk_text = ""
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        test_chunk = current_chunk_text + '\n' + paragraph if current_chunk_text else paragraph
        if estimate_tokens(test_chunk) > max_tokens and current_chunk_text:
            chunks.append(current_chunk_text)
            current_chunk = [paragraph]
            current_chunk_text = paragraph
        else:
            current_chunk.append(paragraph)
            current_chunk_text = test_chunk
    
    if current_chunk_text:
        chunks.append(current_chunk_text)
    
    # Further split chunks that exceed max tokens (by sentences and then words)
    final_chunks = []
    for chunk in chunks:
        if estimate_tokens(chunk) <= max_tokens:
            final_chunks.append(chunk)
        else:
            sentences = []
            for sentence in chunk.replace('. ', '.\n').replace('! ', '!\n').replace('? ', '?\n').split('\n'):
                if sentence.strip():
                    sentences.append(sentence.strip())
            
            sentence_chunk = []
            sentence_chunk_text = ""
            for sentence in sentences:
                test_sentence_chunk = sentence_chunk_text + ' ' + sentence if sentence_chunk_text else sentence
                if estimate_tokens(test_sentence_chunk) > max_tokens and sentence_chunk_text:
                    final_chunks.append(sentence_chunk_text)
                    sentence_chunk = [sentence]
                    sentence_chunk_text = sentence
                else:
                    sentence_chunk.append(sentence)
                    sentence_chunk_text = test_sentence_chunk
            
            if sentence_chunk_text:
                final_chunks.append(sentence_chunk_text)
    
    verified_chunks = []
    for chunk in final_chunks:
        tokens = estimate_tokens(chunk)
        if tokens > max_tokens:
            words = chunk.split()
            small_chunk = []
            small_chunk_text = ""
            for word in words:
                test_word_chunk = small_chunk_text + ' ' + word if small_chunk_text else word
                if estimate_tokens(test_word_chunk) > max_tokens and small_chunk_text:
                    verified_chunks.append(small_chunk_text)
                    small_chunk = [word]
                    small_chunk_text = word
                else:
                    small_chunk.append(word)
                    small_chunk_text = test_word_chunk
            if small_chunk_text:
                verified_chunks.append(small_chunk_text)
        else:
            verified_chunks.append(chunk)
    
    if len(verified_chunks) > MAX_CHUNKS:
        logger.warning(f"Text split into {len(verified_chunks)} chunks, limiting to {MAX_CHUNKS}")
        verified_chunks = verified_chunks[:MAX_CHUNKS]
    
    return verified_chunks

# Function that uses Crew AI for translation and editing
def translate_chunk_with_crew(chunk_text, source_language, target_language, api_key):
    # Set the API key for Crew AI
    os.environ['OPENAI_API_KEY'] = api_key
    
    try:
        # Initialize LLM with timeout
        llm = LLM(model="gpt-3.5-turbo", timeout=120)
        
        # Create agents
        translator = Agent(
            role='Translator',
            goal=f'Translate text from {source_language} to {target_language}',
            backstory='Expert translator',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        
        editor = Agent(
            role='Editor',
            goal=f'Refine {target_language} translation',
            backstory='Experienced editor',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        
        # Define tasks
        translate_task = Task(
            description=f"Translate from {source_language} to {target_language}: {chunk_text}",
            agent=translator,
            expected_output='Translated text'
        )
        
        edit_task = Task(
            description=f"Refine translation for {target_language}.",
            agent=editor,
            expected_output='Polished translation'
        )
        
        # Create and run crew
        translation_crew = Crew(
            agents=[translator, editor],
            tasks=[translate_task, edit_task],
            verbose=True,
            process=Process.sequential
        )
        
        result = translation_crew.kickoff()
        if hasattr(result, 'raw'):
            return result.raw
        return str(result)
    
    except Exception as e:
        raise Exception(f"Crew AI translation error: {str(e)}")

# Get job data from the job store
def get_job_data(job_id):
    with job_lock:
        return jobs.get(job_id)

# Update job data in the job store
def update_job_data(job_id, data):
    with job_lock:
        if job_id in jobs:
            jobs[job_id].update(data)
        else:
            jobs[job_id] = data

# Save job result to file
def save_job_result(job_id, result_data):
    result_file = os.path.join(RESULTS_DIR, f"{job_id}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

# Load job result from file
def load_job_result(job_id):
    result_file = os.path.join(RESULTS_DIR, f"{job_id}.json")
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# Clean up job data
def clean_up_job(job_id):
    with job_lock:
        if job_id in jobs:
            del jobs[job_id]
    result_file = os.path.join(RESULTS_DIR, f"{job_id}.json")
    if os.path.exists(result_file):
        def delete_file():
            try:
                os.remove(result_file)
                logger.info(f"Cleaned up job file for {job_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up job file for {job_id}: {e}")
        t = threading.Timer(3600, delete_file)
        t.daemon = True
        t.start()

# Process a single chunk using Crew AI
def process_chunk(job_id, chunk_idx, chunk, source_language, target_language, api_key):
    logger.info(f"Processing chunk {chunk_idx+1} for job {job_id}")
    
    success = False
    result_text = None
    retry_count = 0
    
    while not success and retry_count < MAX_RETRIES:
        if retry_count > 0:
            logger.info(f"Retry {retry_count} for chunk {chunk_idx+1} of job {job_id}")
            time.sleep(retry_count * 2)
        try:
            result_text = translate_chunk_with_crew(chunk, source_language, target_language, api_key)
            if result_text:
                success = True
                logger.info(f"Chunk {chunk_idx+1} of job {job_id} completed successfully")
            else:
                logger.error(f"No result returned for chunk {chunk_idx+1} of job {job_id}")
                retry_count += 1
        except Exception as e:
            logger.error(f"Error translating chunk {chunk_idx+1} of job {job_id}: {e}")
            retry_count += 1
    
    return {
        "chunk_idx": chunk_idx,
        "success": success,
        "result": result_text
    }

# Process a translation job in the background
def process_translation_job(job_id, source_text, source_language, target_language, api_key):
    try:
        update_job_data(job_id, {"status": "processing"})
        chunks = split_text(source_text, max_tokens=MAX_TOKENS_PER_CHUNK)
        update_job_data(job_id, {"total_chunks": len(chunks)})
        logger.info(f"Job {job_id}: Split into {len(chunks)} chunks")
        
        semaphore = threading.Semaphore(MAX_CONCURRENT_PROCESSES)
        results = [None] * len(chunks)
        failed_chunks = []
        
        def process_chunk_with_semaphore(idx, chunk):
            with semaphore:
                result = process_chunk(job_id, idx, chunk, source_language, target_language, api_key)
                results[idx] = result
                job_data = get_job_data(job_id)
                if job_data:
                    completed = sum(1 for r in results if r is not None and r["success"])
                    job_data["chunks_completed"] = completed
                    job_data["progress"] = completed / len(chunks)
                    update_job_data(job_id, job_data)
                return result
        
        threads = []
        for i, chunk in enumerate(chunks):
            thread = threading.Thread(
                target=process_chunk_with_semaphore,
                args=(i, chunk)
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
            time.sleep(0.1)
        
        for thread in threads:
            thread.join()
        
        all_succeeded = True
        final_results = []
        for i, result in enumerate(results):
            if result is None or not result["success"]:
                all_succeeded = False
                failed_chunks.append(i)
            else:
                final_results.append(result["result"])
        
        if all_succeeded:
            final_translation = " ".join(final_results)
            result_data = {
                "success": True,
                "translation": final_translation,
                "status": "completed",
                "metadata": {
                    "total_chunks": len(chunks),
                    "completion_time": datetime.now().isoformat()
                }
            }
            save_job_result(job_id, result_data)
            update_job_data(job_id, {"status": "completed"})
            logger.info(f"Job {job_id}: Translation completed successfully")
        else:
            if final_results:
                partial_translation = " ".join(final_results)
                result_data = {
                    "success": False,
                    "partial_translation": partial_translation,
                    "failed_chunks": failed_chunks,
                    "status": "failed",
                    "error": f"Failed to process {len(failed_chunks)} out of {len(chunks)} chunks",
                    "metadata": {
                        "total_chunks": len(chunks),
                        "completed_chunks": len(final_results),
                        "completion_time": datetime.now().isoformat()
                    }
                }
            else:
                result_data = {
                    "success": False,
                    "status": "failed",
                    "error": "Failed to process any chunks successfully",
                    "metadata": {
                        "total_chunks": len(chunks),
                        "completed_chunks": 0,
                        "completion_time": datetime.now().isoformat()
                    }
                }
            save_job_result(job_id, result_data)
            update_job_data(job_id, {"status": "failed"})
            logger.error(f"Job {job_id}: Translation failed for chunks {failed_chunks}")
    
    except Exception as e:
        logger.exception(f"Job {job_id}: Error processing translation job")
        result_data = {
            "success": False,
            "status": "failed",
            "error": str(e),
            "metadata": {
                "completion_time": datetime.now().isoformat()
            }
        }
        save_job_result(job_id, result_data)
        update_job_data(job_id, {"status": "failed"})
    
    threading.Timer(3600, lambda: clean_up_job(job_id)).start()  # Clean up after 1 hour

# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Translation service is running",
        "version": "3.0"
    })

# Start translation job endpoint
@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        required_fields = ['sourceText', 'sourceLanguage', 'targetLanguage', 'openaiApiKey']
        if not all(data.get(field) for field in required_fields):
            logger.error("Missing required fields in request")
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        source_text = data.get('sourceText')
        source_language = data.get('sourceLanguage')
        target_language = data.get('targetLanguage')
        openai_api_key = data.get('openaiApiKey')
        
        total_tokens = estimate_tokens(source_text)
        estimated_cost = estimate_cost(total_tokens)
        
        if total_tokens > MAX_TOTAL_TOKENS:
            logger.error(f"Input too large: {total_tokens} tokens")
            return jsonify({
                'success': False, 
                'error': f'Text too large for processing. Maximum allowed tokens: {MAX_TOTAL_TOKENS}',
                'estimated_cost': f"${estimated_cost:.4f}"
            }), 400
        
        job_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "total_tokens": total_tokens,
            "estimated_cost": estimated_cost,
            "source_language": source_language,
            "target_language": target_language,
            "chunks_completed": 0,
            "progress": 0.0
        }
        update_job_data(job_id, job_data)
        
        thread = threading.Thread(
            target=process_translation_job,
            args=(job_id, source_text, source_language, target_language, openai_api_key)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'status': 'created',
            'estimated_cost': f"${estimated_cost:.4f}"
        })
        
    except Exception as e:
        logger.exception("Error starting translation job")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Job status endpoint
@app.route('/job/<job_id>', methods=['GET'])
def job_status(job_id):
    try:
        result = load_job_result(job_id)
        if result:
            return jsonify(result)
        job_data = get_job_data(job_id)
        if job_data:
            return jsonify({
                'success': True,
                'job_id': job_id,
                'status': job_data.get('status', 'unknown'),
                'progress': job_data.get('progress', 0),
                'chunks_completed': job_data.get('chunks_completed', 0),
                'total_chunks': job_data.get('total_chunks', 0),
                'estimated_cost': f"${job_data.get('estimated_cost', 0):.4f}"
            })
        return jsonify({
            'success': False,
            'error': 'Job not found'
        }), 404
        
    except Exception as e:
        logger.exception(f"Error getting job status for {job_id}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Cancel job endpoint
@app.route('/job/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    try:
        job_data = get_job_data(job_id)
        if not job_data:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404
        update_job_data(job_id, {'status': 'cancelled'})
        result_data = {
            'success': False,
            'status': 'cancelled',
            'error': 'Job cancelled by user',
            'metadata': {
                'cancellation_time': datetime.now().isoformat()
            }
        }
        save_job_result(job_id, result_data)
        return jsonify({
            'success': True,
            'job_id': job_id,
            'status': 'cancelled'
        })
    except Exception as e:
        logger.exception(f"Error cancelling job {job_id}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Clean up on exit
def cleanup_on_exit(*args, **kwargs):
    logger.info("Cleaning up before shutdown")
    with job_lock:
        for job_id, job_data in jobs.items():
            if job_data.get('status') in ['created', 'processing']:
                job_data['status'] = 'interrupted'
                result_data = {
                    'success': False,
                    'status': 'interrupted',
                    'error': 'Service shutdown before job completion',
                    'metadata': {
                        'interruption_time': datetime.now().isoformat()
                    }
                }
                save_job_result(job_id, result_data)
    sys.exit(0)

signal.signal(signal.SIGTERM, cleanup_on_exit)
signal.signal(signal.SIGINT, cleanup_on_exit)

# Maintenance thread to clean up old jobs
def maintenance_thread():
    while True:
        try:
            now = time.time()
            for filename in os.listdir(RESULTS_DIR):
                filepath = os.path.join(RESULTS_DIR, filename)
                if os.path.isfile(filepath) and now - os.path.getmtime(filepath) > 86400:
                    try:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old result file: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up old result file {filename}: {e}")
        except Exception as e:
            logger.error(f"Error in maintenance thread: {e}")
        time.sleep(3600)

maintenance = threading.Thread(target=maintenance_thread)
maintenance.daemon = True
maintenance.start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    try:
        for item in os.listdir(TEMP_DIR):
            item_path = os.path.join(TEMP_DIR, item)
            if os.path.isdir(item_path) and item != "results":
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                os.remove(item_path)
    except Exception as e:
        logger.warning(f"Error cleaning temp directory: {e}")
    
    os.environ["WEB_CONCURRENCY"] = "1"
    os.environ["GUNICORN_CMD_ARGS"] = "--timeout 180 --graceful-timeout 60 --keep-alive 5 --max-requests 1000 --max-requests-jitter 50"
    
    app.run(host='0.0.0.0', port=port, threaded=True)
