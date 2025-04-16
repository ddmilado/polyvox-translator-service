from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import subprocess
import uuid
import json
import time
from datetime import datetime
import tiktoken
import logging
import tempfile
import shutil
import psutil
import threading
import queue
import signal
from functools import partial

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
MAX_TOKENS_PER_CHUNK = 1000  # Further reduced chunk size
MAX_TOTAL_TOKENS = 100000
TOKEN_COST_PER_1K = 0.01
PROCESS_TIMEOUT = 120  # Increased timeout (2 minutes)
MAX_RETRIES = 3  # Allow retrying failed chunks
TEMP_DIR = os.path.join(tempfile.gettempdir(), "translation_service")
MAX_CONCURRENT_PROCESSES = 2  # Limit concurrent translation processes
MEMORY_LIMIT_PERCENT = 75  # Kill process if memory usage exceeds this percentage

# Create temp directory if it doesn't exist
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Function to estimate token count
def estimate_tokens(text, model="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error estimating tokens: {e}")
        # Fallback to character-based estimation (rough approximation)
        return len(text) // 4

# Function to estimate translation cost
def estimate_cost(total_tokens):
    return (total_tokens / 1000) * TOKEN_COST_PER_1K

# Function to split text into smaller chunks with improved handling
def split_text(text, max_tokens=MAX_TOKENS_PER_CHUNK, model="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return [text]
            
        # Enhanced text splitting strategy
        chunks = []
        
        # First try to split by paragraphs
        paragraphs = text.split('\n')
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_tokens = len(encoding.encode(paragraph))
            
            # If a single paragraph exceeds max tokens, split it by sentences
            if paragraph_tokens > max_tokens:
                # Add the current chunk if not empty
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split the paragraph into sentences
                sentences = []
                for sent in paragraph.replace('. ', '.\n').replace('! ', '!\n').replace('? ', '?\n').split('\n'):
                    if sent.strip():
                        sentences.append(sent.strip())
                
                # Process sentences
                sentence_chunk = []
                sentence_length = 0
                
                for sentence in sentences:
                    sentence_tokens = len(encoding.encode(sentence))
                    
                    # If a single sentence exceeds max tokens, split it directly
                    if sentence_tokens > max_tokens:
                        # Add current sentence chunk if not empty
                        if sentence_chunk:
                            chunks.append(' '.join(sentence_chunk))
                            sentence_chunk = []
                            sentence_length = 0
                        
                        # Split the sentence by tokens
                        sentence_tokens_list = encoding.encode(sentence)
                        for i in range(0, len(sentence_tokens_list), max_tokens // 2):
                            sub_tokens = sentence_tokens_list[i:i + (max_tokens // 2)]
                            sub_text = encoding.decode(sub_tokens)
                            chunks.append(sub_text)
                    
                    # Check if adding this sentence would exceed the limit
                    elif sentence_length + sentence_tokens > max_tokens:
                        chunks.append(' '.join(sentence_chunk))
                        sentence_chunk = [sentence]
                        sentence_length = sentence_tokens
                    else:
                        sentence_chunk.append(sentence)
                        sentence_length += sentence_tokens
                
                # Add the last sentence chunk if it exists
                if sentence_chunk:
                    chunks.append(' '.join(sentence_chunk))
            
            # Check if adding this paragraph would exceed the limit
            elif current_length + paragraph_tokens > max_tokens:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = paragraph_tokens
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_tokens
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # Verify that no chunk exceeds the token limit
        final_chunks = []
        for chunk in chunks:
            chunk_tokens = len(encoding.encode(chunk))
            if chunk_tokens > max_tokens:
                # This shouldn't happen with our approach, but just in case
                chunk_tokens_list = encoding.encode(chunk)
                for i in range(0, len(chunk_tokens_list), max_tokens // 2):
                    sub_tokens = chunk_tokens_list[i:i + (max_tokens // 2)]
                    sub_text = encoding.decode(sub_tokens)
                    final_chunks.append(sub_text)
            else:
                final_chunks.append(chunk)
        
        # Check if we have too many chunks
        if len(final_chunks) > MAX_CHUNKS:
            logger.warning(f"Text split into {len(final_chunks)} chunks, limiting to {MAX_CHUNKS}")
            final_chunks = final_chunks[:MAX_CHUNKS]
        
        return final_chunks
        
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        # Fallback to a simpler approach
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            if len(' '.join(current_chunk + [word])) > max_tokens * 3:  # Rough character estimation
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

# Write simpler translator script that uses only one agent
def write_simplified_translator_script():
    script_path = os.path.join(TEMP_DIR, "simplified_translator.py")
    with open(script_path, 'w') as f:
        f.write("""
import sys
import json
import os
import traceback
from openai import OpenAI

# Get arguments
chunk_text = sys.argv[1]
source_language = sys.argv[2]
target_language = sys.argv[3]
api_key = sys.argv[4]
output_file = sys.argv[5]

try:
    # Set up OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Create system message
    system_message = f"You are a professional translator. Translate the following text from {source_language} to {target_language}. Keep the meaning, tone, and style intact."
    
    # Make API call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": chunk_text}
        ],
        temperature=0.3,
        max_tokens=2048
    )
    
    # Get translation
    translation = response.choices[0].message.content
    
    # Write result to output file
    with open(output_file, 'w') as f:
        f.write(translation)
    
    # Success exit code
    sys.exit(0)
except Exception as e:
    # Write error to output file
    with open(output_file, 'w') as f:
        f.write(f"ERROR: {str(e)}\\n{traceback.format_exc()}")
    
    # Error exit code
    sys.exit(1)
""")
    return script_path

# Process management class
class TranslationProcessor:
    def __init__(self):
        self.process_semaphore = threading.Semaphore(MAX_CONCURRENT_PROCESSES)
        self.active_processes = {}
        self.process_lock = threading.Lock()
    
    def run_subprocess_safe(self, cmd, timeout=PROCESS_TIMEOUT):
        """Run subprocess with timeout and memory monitoring"""
        with self.process_semaphore:
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Register process
                with self.process_lock:
                    self.active_processes[process.pid] = process
                
                # Start memory monitoring
                stop_monitoring = threading.Event()
                monitor_thread = threading.Thread(
                    target=self._monitor_process_memory,
                    args=(process.pid, stop_monitoring)
                )
                monitor_thread.daemon = True
                monitor_thread.start()
                
                try:
                    # Wait for process with timeout
                    stdout, stderr = process.communicate(timeout=timeout)
                    
                    # Stop monitoring
                    stop_monitoring.set()
                    
                    # Unregister process
                    with self.process_lock:
                        if process.pid in self.active_processes:
                            del self.active_processes[process.pid]
                    
                    return process.returncode, stdout, stderr
                    
                except subprocess.TimeoutExpired:
                    logger.warning(f"Process {process.pid} timed out after {timeout}s")
                    self._kill_process_tree(process.pid)
                    return -1, "", "Process timed out"
                    
            except Exception as e:
                logger.error(f"Error running subprocess: {e}")
                return -1, "", str(e)
    
    def _monitor_process_memory(self, pid, stop_event):
        """Monitor process memory usage and kill if it exceeds limits"""
        try:
            while not stop_event.is_set():
                try:
                    proc = psutil.Process(pid)
                    mem_info = proc.memory_info()
                    mem_percent = mem_info.rss / psutil.virtual_memory().total * 100
                    
                    if mem_percent > MEMORY_LIMIT_PERCENT:
                        logger.warning(f"Process {pid} using {mem_percent:.1f}% memory, killing")
                        self._kill_process_tree(pid)
                        break
                except (psutil.NoSuchProcess, ProcessLookupError):
                    break  # Process ended
                    
                # Check every 0.5 seconds
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error monitoring process memory: {e}")
    
    def _kill_process_tree(self, pid):
        """Kill a process and all its children"""
        try:
            with self.process_lock:
                if pid in self.active_processes:
                    parent = psutil.Process(pid)
                    children = parent.children(recursive=True)
                    
                    # Kill children
                    for child in children:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                            
                    # Kill parent
                    try:
                        parent.kill()
                    except psutil.NoSuchProcess:
                        pass
                        
                    # Remove from active processes
                    del self.active_processes[pid]
                    
        except Exception as e:
            logger.error(f"Error killing process tree: {e}")
    
    def clean_up(self):
        """Kill all active processes"""
        with self.process_lock:
            for pid in list(self.active_processes.keys()):
                self._kill_process_tree(pid)

# Create processor instance
translation_processor = TranslationProcessor()

# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Translation service is running",
        "version": "2.0"
    })

# Translation endpoint
@app.route('/translate', methods=['POST'])
def translate():
    # Create a session ID for this translation job
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    try:
        # Parse request data
        data = request.json
        required_fields = ['sourceText', 'sourceLanguage', 'targetLanguage', 'openaiApiKey']
        if not all(data.get(field) for field in required_fields):
            logger.error("Missing required fields in request")
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Extract data from request
        source_text = data.get('sourceText')
        source_language = data.get('sourceLanguage')
        target_language = data.get('targetLanguage')
        openai_api_key = data.get('openaiApiKey')
        
        # Estimate total tokens and log
        total_tokens = estimate_tokens(source_text)
        estimated_cost = estimate_cost(total_tokens)
        logger.info(f"Session {session_id}: Total estimated tokens: {total_tokens}, Estimated cost: ${estimated_cost:.4f}")
        
        # Prevent processing of excessively large inputs
        if total_tokens > MAX_TOTAL_TOKENS:
            logger.error(f"Session {session_id}: Input too large: {total_tokens} tokens")
            return jsonify({
                'success': False, 
                'error': f'Text too large for processing. Maximum allowed tokens: {MAX_TOTAL_TOKENS}',
                'estimated_cost': f"${estimated_cost:.4f}"
            }), 400
        
        # Split text into smaller chunks
        try:
            chunks = split_text(source_text, max_tokens=MAX_TOKENS_PER_CHUNK)
        except Exception as e:
            logger.error(f"Session {session_id}: Error splitting text: {e}")
            return jsonify({'success': False, 'error': f"Error splitting text: {str(e)}"}), 400
            
        logger.info(f"Session {session_id}: Number of chunks: {len(chunks)}")
        
        # Create simplified script file for translation subprocess
        script_path = write_simplified_translator_script()
        
        # Process each chunk with pool of workers
        results = [None] * len(chunks)  # Pre-allocate results array
        failed_chunks = []
        
        start_time = time.time()
        
        # Process chunks with retries
        for i, chunk in enumerate(chunks):
            logger.info(f"Session {session_id}: Processing chunk {i+1}/{len(chunks)}")
            
            # Retry logic
            success = False
            retry_count = 0
            
            while not success and retry_count < MAX_RETRIES:
                if retry_count > 0:
                    logger.info(f"Session {session_id}: Retry {retry_count} for chunk {i+1}")
                    # Add increasing delay between retries
                    time.sleep(retry_count * 2)
                
                # Save chunk to temporary file
                chunk_file = os.path.join(session_dir, f"chunk_{i}.txt")
                with open(chunk_file, 'w') as f:
                    f.write(chunk)
                
                # Output file for translation result
                output_file = os.path.join(session_dir, f"result_{i}.txt")
                
                # Create command for separate process using the simplified translator
                cmd = [
                    "python3", script_path,
                    chunk, source_language, target_language, 
                    openai_api_key, output_file
                ]
                
                # Run translation in separate process with timeout
                logger.info(f"Session {session_id}: Starting subprocess for chunk {i+1}")
                returncode, stdout, stderr = translation_processor.run_subprocess_safe(cmd, timeout=PROCESS_TIMEOUT)
                
                # Check if translation succeeded
                if returncode == 0 and os.path.exists(output_file):
                    try:
                        with open(output_file, 'r') as f:
                            result = f.read()
                            
                        if result.startswith("ERROR:"):
                            logger.error(f"Session {session_id}: Chunk {i+1} failed: {result}")
                            retry_count += 1
                        else:
                            results[i] = result
                            logger.info(f"Session {session_id}: Chunk {i+1} completed successfully")
                            success = True
                    except Exception as e:
                        logger.error(f"Session {session_id}: Error reading result file: {e}")
                        retry_count += 1
                else:
                    logger.error(f"Session {session_id}: Chunk {i+1} failed with return code {returncode}")
                    logger.error(f"Session {session_id}: STDERR: {stderr}")
                    retry_count += 1
            
            # If all retries failed, mark this chunk as failed
            if not success:
                failed_chunks.append(i)
                logger.error(f"Session {session_id}: All retries failed for chunk {i+1}")
            
            # Add a small delay to avoid rate limits but only if not the last chunk
            if i < len(chunks) - 1:
                time.sleep(0.5)
        
        # Clean up session directory
        try:
            shutil.rmtree(session_dir)
        except Exception as e:
            logger.warning(f"Session {session_id}: Failed to clean up session directory: {e}")
        
        # Check if any chunks failed
        if failed_chunks:
            logger.error(f"Session {session_id}: Failed to process chunks: {failed_chunks}")
            
            # Return partial results if we have any
            valid_results = [r for r in results if r is not None]
            if valid_results:
                partial_result = " ".join(valid_results)
                return jsonify({
                    'success': False,
                    'error': f"Failed to process {len(failed_chunks)} out of {len(chunks)} chunks",
                    'partial_translation': partial_result,
                    'failed_chunks': failed_chunks,
                    'progress': f"{len(valid_results)}/{len(chunks)} chunks completed",
                    'estimated_cost': f"${estimate_cost(total_tokens * (len(valid_results)/len(chunks))):.4f}"
                }), 500
            else:
                return jsonify({
                    'success': False,
                    'error': "Failed to process any chunks successfully",
                    'estimated_cost': f"${estimated_cost:.4f}"
                }), 500
        
        # Combine all results
        final_translation = " ".join(results)
        total_time = time.time() - start_time
        
        logger.info(f"Session {session_id}: Translation completed in {total_time:.2f} seconds")
        
        # Return successful response
        return jsonify({
            'success': True,
            'translation': final_translation,
            'metadata': {
                'total_chunks': len(chunks),
                'processing_time': f"{total_time:.2f} seconds",
                'estimated_cost': f"${estimated_cost:.4f}",
                'completion_time': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        logger.exception(f"Session {session_id}: Translation failed")
        
        # Clean up session directory
        try:
            shutil.rmtree(session_dir)
        except:
            pass
            
        return jsonify({
            'success': False,
            'error': str(e),
            'estimated_cost': f"${estimated_cost:.4f}" if 'estimated_cost' in locals() else None
        }), 500

# Graceful shutdown
def cleanup_on_exit(*args, **kwargs):
    logger.info("Cleaning up before shutdown")
    translation_processor.clean_up()
    
    # Clean up any remaining temp files
    try:
        for item in os.listdir(TEMP_DIR):
            item_path = os.path.join(TEMP_DIR, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    except Exception as e:
        logger.warning(f"Error cleaning temp directory: {e}")

# Register signal handlers
signal.signal(signal.SIGTERM, cleanup_on_exit)
signal.signal(signal.SIGINT, cleanup_on_exit)

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    
    # Clean up any temp files from previous runs
    try:
        for item in os.listdir(TEMP_DIR):
            item_path = os.path.join(TEMP_DIR, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                os.remove(item_path)
    except Exception as e:
        logger.warning(f"Error cleaning temp directory: {e}")
    
    # Set Gunicorn configuration via environment variables
    os.environ["WEB_CONCURRENCY"] = "1"  # Limit to single worker
    os.environ["GUNICORN_CMD_ARGS"] = "--timeout 120 --graceful-timeout 30 --keep-alive 5 --max-requests 100 --max-requests-jitter 20"
    
    app.run(host='0.0.0.0', port=port, threaded=True)