from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from crewai import Agent, Task, Crew, Process, LLM
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Validate input size to prevent overload
        if len(source_text) > 55000:  # Adjust limit based on testing
            logger.error("Input text too long")
            return jsonify({'success': False, 'error': 'Text exceeds maximum length'}), 400
        
        os.environ['OPENAI_API_KEY'] = openai_api_key
        llm = LLM(model="gpt-3.5-turbo")  # Lighter model to reduce resource usage
        
        # Define translation agent
        translator = Agent(
            role='Translator',
            goal='Translate text accurately while preserving context and meaning',
            backstory='Expert translator with deep knowledge of languages and cultural nuances.',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        
        # Define editor agent
        editor = Agent(
            role='Editor',
            goal='Refine translations to ensure they are natural and idiomatic',
            backstory='Professional editor experienced in polishing translations.',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
        
        # Define translation task
        translate_task = Task(
            description=f"Translate this text from {source_language} to {target_language}: {source_text}",
            agent=translator,
            expected_output='Translated text preserving original meaning and tone'
        )
        
        # Define editing task
        edit_task = Task(
            description=f"Refine the translated text to sound natural in {target_language}, fixing awkward phrasing or errors.",
            agent=editor,
            expected_output='Polished translation that reads naturally'
        )
        
        # Create crew
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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Default to 10000 for Render
    app.run(host='0.0.0.0', port=port)

# Optional: For asynchronous processing with Celery (uncomment and configure if needed)
"""
from celery import Celery

celery_app = Celery('tasks', broker='redis://redis:6379/0')

@celery_app.task
def run_translation(source_text, source_language, target_language, openai_api_key):
    os.environ['OPENAI_API_KEY'] = openai_api_key
    llm = LLM(model="gpt-3.5-turbo")
    # ... (same agent/task setup as above)
    return translation_crew.kickoff()

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        if not all([data.get('sourceText'), data.get('sourceLanguage'), data.get('targetLanguage'), data.get('openaiApiKey')]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        task = run_translation.delay(data.get('sourceText'), data.get('sourceLanguage'), data.get('targetLanguage'), data.get('openaiApiKey'))
        return jsonify({'success': True, 'task_id': task.id, 'message': 'Translation is being processed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/task/<task_id>', methods=['GET'])
def task_status(task_id):
    task = run_translation.AsyncResult(task_id)
    if task.state == 'PENDING':
        return jsonify({'success': True, 'state': 'PENDING'})
    elif task.state == 'SUCCESS':
        return jsonify({'success': True, 'state': 'SUCCESS', 'translation': task.get()})
    else:
        return jsonify({'success': False, 'state': task.state})
"""