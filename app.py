from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

app = Flask(__name__)
CORS(app)

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    
    # Extract data from request
    source_text = data.get('sourceText')
    source_language = data.get('sourceLanguage')
    target_language = data.get('targetLanguage')
    openai_api_key = data.get('openaiApiKey')
    
    # Set OpenAI API key
    os.environ['OPENAI_API_KEY'] = openai_api_key
    
    # Initialize OpenAI client
    openai = ChatOpenAI(model="gpt-4")
    
    # Define translation agents
    translator = Agent(
        name='Translator',
        goal='Translate documents with perfect accuracy while maintaining context and meaning',
        backstory='You are an expert translator with deep knowledge of multiple languages and cultural nuances.',
        verbose=True,
        allow_delegation=False,
        llm=openai
    )
    
    editor = Agent(
        name='Editor',
        goal='Review and refine translations to ensure they are natural and idiomatic',
        backstory='You are a professional editor with years of experience in refining translations.',
        verbose=True,
        allow_delegation=False,
        llm=openai
    )
    
    # Define translation task
    translate_task = Task(
        description=f"Translate the following text from {source_language} to {target_language}. "
                   f"Maintain the original formatting, tone, and meaning. "
                   f"The text to translate is: {source_text}",
        agent=translator,
        expected_output='The translated text with original formatting preserved'
    )
    
    edit_task = Task(
        description=f"Review and refine the translation to ensure it sounds natural in {target_language}. "
                   f"Fix any awkward phrasing, grammatical errors, or literal translations that don't capture the intended meaning.",
        agent=editor,
        expected_output='The refined translation that reads naturally in the target language'
    )
    
    # Create crew
    translation_crew = Crew(
        agents=[translator, editor],
        tasks=[translate_task, edit_task],
        verbose=True
    )
    
    # Execute the translation crew
    result = translation_crew.run()
    
    return jsonify({
        'success': True,
        'translation': result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))