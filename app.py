from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from crewai import Agent, Task, Crew, Process, LLM
import logging
import tiktoken
import time
from datetime import datetime
import json

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_CHUNKS = 50  # Maximum number of chunks to process
MAX_TOKENS_PER_CHUNK = 10000
MAX_TOTAL_TOKENS = 100000
TOKEN_COST_PER_1K = 0.01  # Cost per 1K tokens for GPT-4-turbo-preview

# Function to estimate token count using tiktoken
def estimate_tokens(text, model="gpt-4-turbo-preview"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Function to estimate translation cost
def estimate_cost(total_tokens):
    return (total_tokens / 1000) * TOKEN_COST_PER_1K

# Function to split text into chunks based on token limit
def split_text(text, max_tokens=MAX_TOKENS_PER_CHUNK, model="gpt-4-turbo-preview"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    # If the entire text is smaller than max_tokens, return it as a single chunk
    if len(tokens) <= max_tokens:
        return [text]
    
    # Calculate minimum chunk size based on the number of tokens
    min_chunk_size = len(tokens) // 4  # Start with 4 chunks
    if min_chunk_size > max_tokens:
        # If even 4 chunks would be too large, calculate the minimum number of chunks needed
        min_chunks = (len(tokens) // max_tokens) + 1
        if min_chunks > MAX_CHUNKS:
            raise ValueError(f"Text requires {min_chunks} chunks, which exceeds maximum allowed chunks ({MAX_CHUNKS})")
        chunk_size = len(tokens) // min_chunks
    else:
        chunk_size = min_chunk_size
    
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Translation service is running"})

# Translation endpoint
@app.route('/translate', methods=['POST'])
def translate():
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
        
        # Set OpenAI API key and initialize LLM
        os.environ['OPENAI_API_KEY'] = openai_api_key
        llm = LLM(model="gpt-4-turbo-preview")
        
        # Define translator agent
        translator = Agent(
            role='Translator',
            goal='Translate text accurately',
            backstory='Expert translator',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        
        # Define editor agent
        editor = Agent(
            role='Editor',
            goal='Refine translations',
            backstory='Experienced editor',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        
        # Split text into manageable chunks
        try:
            chunks = split_text(source_text, max_tokens=MAX_TOKENS_PER_CHUNK)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
            
        logger.info(f"Number of chunks: {len(chunks)}")
        
        # Verify that no chunk exceeds the maximum token limit
        for i, chunk in enumerate(chunks):
            chunk_tokens = estimate_tokens(chunk)
            if chunk_tokens > MAX_TOKENS_PER_CHUNK:
                logger.error(f"Chunk {i+1} exceeds maximum token limit: {chunk_tokens} tokens")
                return jsonify({
                    'success': False, 
                    'error': f'Unable to process text: chunk {i+1} exceeds maximum token limit',
                    'estimated_cost': f"${estimated_cost:.4f}"
                }), 400
        
        # Process each chunk and collect results
        final_translation = []
        start_time = time.time()
        
        for i, chunk in enumerate(chunks):
            try:
                chunk_start_time = time.time()
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Define translation task
                translate_task = Task(
                    description=f"Translate from {source_language} to {target_language}: {chunk}",
                    agent=translator,
                    expected_output='Translated text'
                )
                
                # Define editing task
                edit_task = Task(
                    description=f"Refine translation for {target_language}.",
                    agent=editor,
                    expected_output='Polished translation'
                )
                
                # Create and run crew for this chunk
                translation_crew = Crew(
                    agents=[translator, editor],
                    tasks=[translate_task, edit_task],
                    verbose=True,
                    process=Process.sequential
                )
                
                chunk_result = translation_crew.kickoff()
                
                # Extract the translated text from CrewOutput
                if hasattr(chunk_result, 'raw'):
                    final_translation.append(chunk_result.raw)
                else:
                    logger.error(f"Unable to extract result from CrewOutput for chunk {i+1}")
                    return jsonify({
                        'success': False, 
                        'error': 'Failed to extract translation result',
                        'progress': f"{i}/{len(chunks)} chunks completed",
                        'estimated_cost': f"${estimate_cost(total_tokens * (i/len(chunks))):.4f}"
                    }), 500
                
                chunk_time = time.time() - chunk_start_time
                logger.info(f"Chunk {i+1} completed in {chunk_time:.2f} seconds")
                
                # Add a small delay to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                if "rate_limit_exceeded" in str(e).lower():
                    logger.warning("Rate limit exceeded, waiting before retry...")
                    time.sleep(5)  # Wait 5 seconds before retrying
                    continue
                raise
        
        total_time = time.time() - start_time
        logger.info(f"Translation completed in {total_time:.2f} seconds")
        
        # Combine chunk results into final translation
        result = " ".join(final_translation)
        logger.info("Translation and editing completed successfully")
        
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

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)