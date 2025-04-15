from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from crewai import Agent, Task, Crew, Process, LLM
import logging
import math

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Approximate token estimation (rough, as exact tokenization depends on the model)
def estimate_tokens(text):
    """Estimate the number of tokens in a text string (approx. 4 chars per token)."""
    return len(text) // 4 + len(text.split())  # Accounts for words and punctuation

def split_text(text, max_tokens=12000):
    """Split text into chunks that fit within max_tokens (leaving room for prompts)."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0
    for word in words:
        word_tokens = estimate_tokens(word)
        if current_tokens + word_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_tokens = word_tokens
        else:
            current_chunk.append(word)
            current_tokens += word_tokens
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Translation service is running"})

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        if not all([data.get('sourceText'), data.get('sourceLanguage'), data.get('targetLanguage'), data.get('openaiApiKey')]):
            logger.error("Missing required fields in request")
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        source_text = data.get('sourceText')
        source_language = data.get('sourceLanguage')
        target_language = data.get('targetLanguage')
        openai_api_key = data.get('openaiApiKey')
        
        # Estimate tokens and reject if too large overall
        total_tokens = estimate_tokens(source_text)
        if total_tokens > 100000:  # Arbitrary large limit to prevent abuse
            logger.error(f"Input too large: {total_tokens} tokens")
            return jsonify({'success': False, 'error': 'Text too large for processing'}), 400
        
        os.environ['OPENAI_API_KEY'] = openai_api_key
        llm = LLM(model="gpt-3.5-turbo")  # Lighter model
        
        # Define translation agent
        translator = Agent(
            role='Translator',
            goal='Translate text accurately while preserving meaning',
            backstory='Expert translator with knowledge of languages and nuances.',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        
        # Define editor agent
        editor = Agent(
            role='Editor',
            goal='Refine translations to sound natural',
            backstory='Experienced editor for polished translations.',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        
        # Check if chunking is needed
        if total_tokens > 12000:  # Leave room for prompts (~4000 tokens)
            logger.info(f"Chunking text: {total_tokens} tokens")
            chunks = split_text(source_text, max_tokens=12000)
            final_translation = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Define tasks for each chunk
                translate_task = Task(
                    description=f"Translate this text from {source_language} to {target_language}: {chunk}",
                    agent=translator,
                    expected_output='Translated text preserving meaning'
                )
                
                edit_task = Task(
                    description=f"Refine the translated text to sound natural in {target_language}.",
                    agent=editor,
                    expected_output='Polished translation'
                )
                
                # Create crew for chunk
                translation_crew = Crew(
                    agents=[translator, editor],
                    tasks=[translate_task, edit_task],
                    verbose=True,
                    process=Process.sequential
                )
                
                chunk_result = translation_crew.kickoff()
                final_translation.append(chunk_result)
            
            result = " ".join(final_translation)
        else:
            # Process as single task if small enough
            translate_task = Task(
                description=f"Translate this text from {source_language} to {target_language}: {source_text}",
                agent=translator,
                expected_output='Translated text preserving meaning'
            )
            
            edit_task = Task(
                description=f"Refine the translated text to sound natural in {target_language}.",
                agent=editor,
                expected_output='Polished translation'
            )
            
            translation_crew = Crew(
                agents=[translator, editor],
                tasks=[translate_task, edit_task],
                verbose=True,
                process=Process.sequential
            )
            
            logger.info("Starting translation crew")
            result = translation_crew.kickoff()
        
        logger.info("Translation and editing completed successfully")
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)