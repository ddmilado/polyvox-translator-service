from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from crewai import Agent, Task, Crew, Process, LLM
import logging
import tiktoken
import time
from datetime import datetime
import json
import gc  # For garbage collection
import psutil  # For memory monitoring
import resource  # For setting memory limits
import threading
import weakref

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_CHUNKS = 50  # Maximum number of chunks to process
MAX_TOKENS_PER_CHUNK = 4000  # Further reduced from 8000 to prevent memory issues
MAX_TOTAL_TOKENS = 100000
TOKEN_COST_PER_1K = 0.01  # Cost per 1K tokens for GPT-4-turbo-preview
TIMEOUT_SECONDS = 180  # Reduce timeout for LLM calls
MAX_MEMORY_PERCENT = 80  # Maximum memory usage percentage

# Set process memory limit
def set_memory_limit(percentage=90):
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        total_memory = psutil.virtual_memory().total
        memory_limit = int(total_memory * (percentage / 100.0))
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, hard))
        logger.info(f"Memory limit set to {memory_limit / (1024 * 1024):.2f} MB ({percentage}% of total)")
    except Exception as e:
        logger.warning(f"Failed to set memory limit: {e}")

# Function to estimate token count using tiktoken
def estimate_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Function to estimate translation cost
def estimate_cost(total_tokens):
    return (total_tokens / 1000) * TOKEN_COST_PER_1K

# Function to split text into smaller chunks based on token limit
def split_text(text, max_tokens=MAX_TOKENS_PER_CHUNK, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    # If the entire text is smaller than max_tokens, return it as a single chunk
    if len(tokens) <= max_tokens:
        return [text]
    
    # For large texts, create more smaller chunks
    # Use paragraph or sentence boundaries where possible
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        paragraph_tokens = len(encoding.encode(paragraph))
        
        # If a single paragraph exceeds the limit, split it by sentences
        if paragraph_tokens > max_tokens:
            sentences = paragraph.replace('. ', '.\n').split('\n')
            for sentence in sentences:
                sentence_tokens = len(encoding.encode(sentence))
                
                # Even a sentence is too long, split by token count directly
                if sentence_tokens > max_tokens:
                    for i in range(0, len(encoding.encode(sentence)), max_tokens // 2):
                        chunk_tokens = encoding.encode(sentence)[i:i + (max_tokens // 2)]
                        chunk_text = encoding.decode(chunk_tokens)
                        chunks.append(chunk_text)
                elif current_length + sentence_tokens > max_tokens:
                    # Current chunk would exceed limit, save it and start a new one
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_tokens
                else:
                    # Add sentence to current chunk
                    current_chunk.append(sentence)
                    current_length += sentence_tokens
        elif current_length + paragraph_tokens > max_tokens:
            # Current chunk would exceed limit, save it and start a new one
            chunks.append('\n'.join(current_chunk))
            current_chunk = [paragraph]
            current_length = paragraph_tokens
        else:
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_length += paragraph_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    # Safety check - ensure no chunk exceeds the token limit
    verified_chunks = []
    for chunk in chunks:
        chunk_tokens = len(encoding.encode(chunk))
        if chunk_tokens > max_tokens:
            # If still too large, force split by token count
            sub_chunks = []
            chunk_encoding = encoding.encode(chunk)
            for i in range(0, len(chunk_encoding), max_tokens // 2):
                sub_chunk_tokens = chunk_encoding[i:i + (max_tokens // 2)]
                sub_chunk_text = encoding.decode(sub_chunk_tokens)
                sub_chunks.append(sub_chunk_text)
            verified_chunks.extend(sub_chunks)
        else:
            verified_chunks.append(chunk)
    
    # Check if we have too many chunks
    if len(verified_chunks) > MAX_CHUNKS:
        logger.warning(f"Text split into {len(verified_chunks)} chunks, exceeding maximum of {MAX_CHUNKS}")
        verified_chunks = verified_chunks[:MAX_CHUNKS]
    
    return verified_chunks

# Function to log memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    percent = psutil.virtual_memory().percent
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB (System: {percent}%)")
    return percent

# Function to check if memory usage is too high
def is_memory_critical():
    percent = psutil.virtual_memory().percent
    return percent > MAX_MEMORY_PERCENT

# Function to limit memory usage
def limit_memory():
    if is_memory_critical():
        logger.warning(f"Memory usage critical at {psutil.virtual_memory().percent}%")
        gc.collect()
        if is_memory_critical():
            # More aggressive memory cleanup
            logger.warning("Performing aggressive memory cleanup")
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
            gc.collect()

# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Translation service is running"})

# Function to process chunk in a separate thread with controlled memory
def process_chunk(chunk, i, source_language, target_language, openai_api_key, results, max_retries=2):
    chunk_start_time = time.time()
    logger.info(f"Processing chunk {i+1} in separate thread")
    
    # Set memory limit for this thread
    set_memory_limit(90)  # 90% of available memory
    
    # Set API key in environment for this thread
    os.environ['OPENAI_API_KEY'] = openai_api_key
    
    for attempt in range(max_retries + 1):
        try:
            # Create LLM with timeout
            llm = LLM(model="gpt-3.5-turbo", timeout=TIMEOUT_SECONDS)
            
            # Create fresh agent instances
            translator = Agent(
                role='Translator',
                goal='Translate text accurately',
                backstory='Expert translator',
                verbose=True,
                allow_delegation=False,
                llm=llm
            )
            
            editor = Agent(
                role='Editor',
                goal='Refine translations',
                backstory='Experienced editor',
                verbose=True,
                allow_delegation=False,
                llm=llm
            )
            
            # Use output files to reduce memory usage
            output_file_translator = f"translate_chunk_{i}.txt"
            output_file_editor = f"edited_chunk_{i}.txt"
            
            # Define translation task
            translate_task = Task(
                description=f"Translate from {source_language} to {target_language}: {chunk}",
                agent=translator,
                expected_output='Translated text',
                output_file=output_file_translator
            )
            
            # Define editing task
            edit_task = Task(
                description=f"Refine translation for {target_language}.",
                agent=editor,
                expected_output='Polished translation',
                output_file=output_file_editor
            )
            
            # Create and run crew for this chunk
            translation_crew = Crew(
                agents=[translator, editor],
                tasks=[translate_task, edit_task],
                verbose=True,
                process=Process.sequential,
                max_rpm=3  # Rate limit to prevent overwhelming the API
            )
            
            chunk_result = translation_crew.kickoff()
            
            # Extract the translated text
            result_text = None
            if hasattr(chunk_result, 'raw'):
                result_text = chunk_result.raw
            elif os.path.exists(output_file_editor):
                # If crew output wasn't accessible, try reading from the output file
                with open(output_file_editor, 'r') as f:
                    result_text = f.read()
            
            if result_text:
                results[i] = result_text
                logger.info(f"Successfully processed chunk {i+1}")
            else:
                raise Exception("Failed to extract translation result")
            
            # Clean up output files
            for file_path in [output_file_translator, output_file_editor]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file {file_path}: {e}")
            
            # Explicit cleanup
            del translator
            del editor
            del translate_task
            del edit_task
            del translation_crew
            del chunk_result
            del llm
            
            # Force garbage collection
            gc.collect()
            limit_memory()
            
            chunk_time = time.time() - chunk_start_time
            logger.info(f"Chunk {i+1} completed in {chunk_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.exception(f"Error processing chunk {i+1}, attempt {attempt+1}/{max_retries+1}")
            if attempt < max_retries:
                # Clean up before retry
                gc.collect()
                limit_memory()
                time.sleep(5 * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Failed to process chunk {i+1} after {max_retries+1} attempts")
                results[i] = f"ERROR_PROCESSING_CHUNK: {str(e)}"
                return False

# Translation endpoint
@app.route('/translate', methods=['POST'])
def translate():
    # Set memory limit for the main process
    set_memory_limit()
    
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
        
        # Log initial memory usage
        log_memory_usage()
        
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
            
        # Split text into manageable chunks
        try:
            chunks = split_text(source_text, max_tokens=MAX_TOKENS_PER_CHUNK)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
            
        logger.info(f"Number of chunks: {len(chunks)}")
        
        # Process chunks with thread pool
        results = {}  # Dictionary to store results by chunk index
        threads = []
        
        # Start time for the whole process
        start_time = time.time()
        
        # Process chunks in batches to control memory usage
        batch_size = 2  # Process only 2 chunks at a time
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_threads = []
            
            for i in range(batch_start, batch_end):
                # Create a thread for each chunk in the batch
                thread = threading.Thread(
                    target=process_chunk,
                    args=(chunks[i], i, source_language, target_language, openai_api_key, results)
                )
                thread.daemon = True
                threads.append(thread)
                batch_threads.append(thread)
                thread.start()
            
            # Wait for all threads in this batch to complete
            for thread in batch_threads:
                thread.join()
            
            # Force garbage collection between batches
            gc.collect()
            limit_memory()
            
            # Check memory usage and log
            mem_percent = log_memory_usage()
            if mem_percent > MAX_MEMORY_PERCENT:
                logger.warning(f"High memory usage ({mem_percent}%) after batch. Cleaning up...")
                # More aggressive cleanup if needed
                os.system("sync")  # Flush filesystem buffers
                gc.collect()
                limit_memory()
        
        # Check if any chunks failed
        failed_chunks = [i for i in range(len(chunks)) if i not in results or results[i].startswith("ERROR_PROCESSING_CHUNK")]
        
        if failed_chunks:
            logger.error(f"Failed to process chunks: {failed_chunks}")
            # Try to return partial results if we have any
            successful_chunks = [i for i in range(len(chunks)) if i in results and not results[i].startswith("ERROR_PROCESSING_CHUNK")]
            
            if successful_chunks:
                partial_result = " ".join([results[i] for i in sorted(successful_chunks)])
                return jsonify({
                    'success': False,
                    'error': f"Failed to process {len(failed_chunks)} out of {len(chunks)} chunks",
                    'partial_translation': partial_result,
                    'failed_chunks': failed_chunks,
                    'progress': f"{len(successful_chunks)}/{len(chunks)} chunks completed",
                    'estimated_cost': f"${estimate_cost(total_tokens * (len(successful_chunks)/len(chunks))):.4f}"
                }), 500
            else:
                return jsonify({
                    'success': False,
                    'error': "Failed to process any chunks successfully",
                    'estimated_cost': f"${estimated_cost:.4f}"
                }), 500
        
        # Combine chunk results into final translation, ensuring proper order
        final_translation = []
        for i in range(len(chunks)):
            if i in results:
                final_translation.append(results[i])
        
        result = " ".join(final_translation)
        total_time = time.time() - start_time
        
        logger.info(f"Translation completed in {total_time:.2f} seconds")
        logger.info("Translation and editing completed successfully")
        
        # Final garbage collection
        gc.collect()
        
        # Return successful response with additional metadata
        return jsonify({
            'success': True,
            'translation': result,
            'metadata': {
                'total_chunks': len(chunks),
                'processing_time': f"{total_time:.2f} seconds",
                'estimated_cost': f"${estimated_cost:.4f}",
                'completion_time': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        logger.exception("Translation failed")
        if "context_length_exceeded" in str(e):
            return jsonify({
                'success': False,
                'error': 'Input text exceeds model context limit. Try a shorter text.',
                'estimated_cost': f"${estimated_cost:.4f}" if 'estimated_cost' in locals() else None
            }), 400
        return jsonify({
            'success': False,
            'error': str(e),
            'estimated_cost': f"${estimated_cost:.4f}" if 'estimated_cost' in locals() else None
        }), 500

# Run the Flask app with Gunicorn worker configuration
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    
    # Log system resources
    logger.info(f"Total system memory: {psutil.virtual_memory().total / (1024*1024*1024):.2f} GB")
    logger.info(f"Available system memory: {psutil.virtual_memory().available / (1024*1024*1024):.2f} GB")
    
    # Add environment variable for Gunicorn workers
    os.environ["WEB_CONCURRENCY"] = "1"  # Limit to single worker to avoid memory issues
    
    # Run app with threading enabled
    app.run(host='0.0.0.0', port=port, threaded=True)