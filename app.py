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
import signal
import shutil
import psutil

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_CHUNKS = 50
MAX_TOKENS_PER_CHUNK = 2000  # Drastically reduced chunk size
MAX_TOTAL_TOKENS = 100000
TOKEN_COST_PER_1K = 0.01
PROCESS_TIMEOUT = 60  # Maximum seconds to wait for a chunk translation
TEMP_DIR = os.path.join(tempfile.gettempdir(), "translation_service")

# Create temp directory if it doesn't exist
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Function to estimate token count
def estimate_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Function to estimate translation cost
def estimate_cost(total_tokens):
    return (total_tokens / 1000) * TOKEN_COST_PER_1K

# Function to split text into very small chunks
def split_text(text, max_tokens=MAX_TOKENS_PER_CHUNK, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    # Split text at sentence boundaries when possible
    sentences = []
    for paragraph in text.split('\n'):
        for sentence in paragraph.replace('. ', '.\n').replace('! ', '!\n').replace('? ', '?\n').split('\n'):
            if sentence.strip():
                sentences.append(sentence.strip())
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence))
        
        # Handle very long sentences by splitting them directly
        if sentence_tokens > max_tokens:
            # If we have content in the current chunk, add it first
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split the long sentence by tokens
            sentence_encoding = encoding.encode(sentence)
            for i in range(0, len(sentence_encoding), max_tokens // 2):
                sub_chunk_tokens = sentence_encoding[i:i + (max_tokens // 2)]
                sub_chunk_text = encoding.decode(sub_chunk_tokens)
                chunks.append(sub_chunk_text)
        
        # Check if adding this sentence would exceed the limit
        elif current_length + sentence_tokens > max_tokens:
            # Save current chunk and start a new one with this sentence
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_tokens
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Ensure no chunk exceeds the token limit
    final_chunks = []
    for chunk in chunks:
        chunk_tokens = len(encoding.encode(chunk))
        if chunk_tokens > max_tokens:
            # This shouldn't happen given our chunking approach, but just in case
            chunk_encoding = encoding.encode(chunk)
            for i in range(0, len(chunk_encoding), max_tokens // 2):
                sub_chunk_tokens = chunk_encoding[i:i + (max_tokens // 2)]
                sub_chunk_text = encoding.decode(sub_chunk_tokens)
                final_chunks.append(sub_chunk_text)
        else:
            final_chunks.append(chunk)
    
    # Check if we have too many chunks
    if len(final_chunks) > MAX_CHUNKS:
        logger.warning(f"Text split into {len(final_chunks)} chunks, limiting to {MAX_CHUNKS}")
        final_chunks = final_chunks[:MAX_CHUNKS]
    
    return final_chunks

# Separate process function for translation to isolate memory usage
def translate_chunk(chunk_id, chunk_text, source_language, target_language, api_key):
    """This runs in a completely separate Python process"""
    import os
    import json
    from crewai import Agent, Task, Crew, Process, LLM
    
    # Set the API key
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
        return {"success": True, "translation": result.raw if hasattr(result, 'raw') else str(result)}
    
    except Exception as e:
        return {"success": False, "error": str(e)}

# Write chunk translator script to a file
def write_chunk_translator_script():
    script_path = os.path.join(TEMP_DIR, "chunk_translator.py")
    with open(script_path, 'w') as f:
        f.write("""
import sys
import json
import os
from crewai import Agent, Task, Crew, Process, LLM
import traceback

# Get arguments
chunk_text = sys.argv[1]
source_language = sys.argv[2]
target_language = sys.argv[3]
api_key = sys.argv[4]
output_file = sys.argv[5]

# Set the API key
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
    
    # Write result to output file
    with open(output_file, 'w') as f:
        if hasattr(result, 'raw'):
            f.write(result.raw)
        else:
            f.write(str(result))
    
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

# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Translation service is running"})

# Run subprocess with timeout and memory monitoring
def run_subprocess_safe(cmd, timeout=PROCESS_TIMEOUT):
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Monitor process memory usage
        def check_process_memory():
            try:
                proc = psutil.Process(process.pid)
                while process.poll() is None:
                    mem_info = proc.memory_info()
                    mem_percent = mem_info.rss / psutil.virtual_memory().total * 100
                    if mem_percent > 85:  # Kill if using more than 85% memory
                        logger.warning(f"Process {process.pid} using {mem_percent:.1f}% memory, killing")
                        process.kill()
                        return
                    time.sleep(1)
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass  # Process already ended
        
        # Start memory monitoring in a separate thread
        import threading
        monitor_thread = threading.Thread(target=check_process_memory)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Wait for process with timeout
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return process.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            logger.warning(f"Process {process.pid} timed out after {timeout}s, killing")
            process.kill()
            return -1, "", "Process timed out"
            
    except Exception as e:
        return -1, "", str(e)

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
        logger.info(f"Total estimated tokens: {total_tokens}, Estimated cost: ${estimated_cost:.4f}")
        
        # Prevent processing of excessively large inputs
        if total_tokens > MAX_TOTAL_TOKENS:
            logger.error(f"Input too large: {total_tokens} tokens")
            return jsonify({
                'success': False, 
                'error': f'Text too large for processing. Maximum allowed tokens: {MAX_TOTAL_TOKENS}',
                'estimated_cost': f"${estimated_cost:.4f}"
            }), 400
        
        # Split text into very small chunks
        try:
            chunks = split_text(source_text, max_tokens=MAX_TOKENS_PER_CHUNK)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
            
        logger.info(f"Number of chunks: {len(chunks)}")
        
        # Create script file for translation subprocess
        script_path = write_chunk_translator_script()
        
        # Process each chunk with a separate process
        results = []
        failed_chunks = []
        
        start_time = time.time()
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Save chunk to temporary file
            chunk_file = os.path.join(session_dir, f"chunk_{i}.txt")
            with open(chunk_file, 'w') as f:
                f.write(chunk)
            
            # Output file for translation result
            output_file = os.path.join(session_dir, f"result_{i}.txt")
            
            # Create command for separate process
            cmd = [
                "python3", script_path,
                chunk, source_language, target_language, 
                openai_api_key, output_file
            ]
            
            # Run translation in separate process with timeout
            logger.info(f"Starting subprocess for chunk {i+1}")
            returncode, stdout, stderr = run_subprocess_safe(cmd, timeout=PROCESS_TIMEOUT)
            
            # Check if translation succeeded
            if returncode == 0 and os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    result = f.read()
                    
                if result.startswith("ERROR:"):
                    logger.error(f"Chunk {i+1} failed: {result}")
                    failed_chunks.append(i)
                else:
                    results.append(result)
                    logger.info(f"Chunk {i+1} completed successfully")
            else:
                logger.error(f"Chunk {i+1} failed with return code {returncode}")
                logger.error(f"STDERR: {stderr}")
                failed_chunks.append(i)
            
            # Add a small delay to avoid rate limits
            time.sleep(1)
        
        # Clean up session directory
        try:
            shutil.rmtree(session_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up session directory: {e}")
        
        # Check if any chunks failed
        if failed_chunks:
            logger.error(f"Failed to process chunks: {failed_chunks}")
            
            # Return partial results if we have any
            if results:
                partial_result = " ".join(results)
                return jsonify({
                    'success': False,
                    'error': f"Failed to process {len(failed_chunks)} out of {len(chunks)} chunks",
                    'partial_translation': partial_result,
                    'failed_chunks': failed_chunks,
                    'progress': f"{len(results)}/{len(chunks)} chunks completed",
                    'estimated_cost': f"${estimate_cost(total_tokens * (len(results)/len(chunks))):.4f}"
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
        
        logger.info(f"Translation completed in {total_time:.2f} seconds")
        
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
        logger.exception("Translation failed")
        
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

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    
    # Clean up any temp files from previous runs
    try:
        for item in os.listdir(TEMP_DIR):
            item_path = os.path.join(TEMP_DIR, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    except Exception as e:
        logger.warning(f"Error cleaning temp directory: {e}")
    
    # Add environment variable for Gunicorn workers
    os.environ["WEB_CONCURRENCY"] = "1"  # Limit to single worker
    
    app.run(host='0.0.0.0', port=port)