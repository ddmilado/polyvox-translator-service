from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from crewai import Agent, Task, Crew, Process, LLM
import logging
import tiktoken

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to estimate token count using tiktoken
def estimate_tokens(text, model="gpt-4o-search-preview"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Function to split text into chunks based on token limit
def split_text(text, max_tokens=10000, model="gpt-4o-search-preview"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
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
        logger.info(f"Total estimated tokens: {total_tokens}")
        
        # Prevent processing of excessively large inputs
        if total_tokens > 100000:
            logger.error(f"Input too large: {total_tokens} tokens")
            return jsonify({'success': False, 'error': 'Text too large for processing'}), 400
        
        # Set OpenAI API key and initialize LLM
        os.environ['OPENAI_API_KEY'] = openai_api_key
        llm = LLM(model="gpt-4o-search-preview")
        
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
        chunks = split_text(source_text, max_tokens=10000)
        logger.info(f"Number of chunks: {len(chunks)}")
        
        # Process each chunk and collect results
        final_translation = []
        for i, chunk in enumerate(chunks):
            chunk_tokens = estimate_tokens(chunk)
            logger.info(f"Processing chunk {i+1}/{len(chunks)} with {chunk_tokens} tokens")
            
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
                return jsonify({'success': False, 'error': 'Failed to extract translation result'}), 500
        
        # Log the collected translation parts for debugging
        logger.info(f"Final translation parts: {final_translation}")
        
        # Combine chunk results into final translation
        result = " ".join(final_translation)
        logger.info("Translation and editing completed successfully")
        
        # Return successful response
        return jsonify({
            'success': True,
            'translation': result
        })
    
    except Exception as e:
        logger.exception("Translation failed")
        if "context_length_exceeded" in str(e):
            return jsonify({
                'success': False,
                'error': 'Input text exceeds model context limit. Try a shorter text.'
            }), 400
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)