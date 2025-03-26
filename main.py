from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import logging
import os
import uuid
import json
from app import get_response, get_connection_types, set_connection_config
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize chat history storage (in-memory for now)
chat_histories = {}

# Initialize Flask app
app = Flask(__name__)
# app.secret_key = os.environ.get("SESSION_SECRET", "power-db-assistant-2025")
app.secret_key = os.getenv("SESSION_SECRET")

@app.route('/')
def index():
    """Render the welcome/database selection page."""
    return render_template('welcome.html')

@app.route('/chat')
def chat_interface():
    """Render the main chat interface."""
    # Generate a unique session ID if one doesn't exist
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    # Get the active connection type
    db_connection = session.get('db_connection', {})
    db_type = db_connection.get('db_type', 'graphdb')  # Default to GraphDB
    
    return render_template('index.html', db_type=db_type)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat messages through the LangGraph workflow."""
    try:
        data = request.get_json()
        user_id = data.get('user_id', session.get('user_id', 'default'))
        user_message = data.get('message', '')
        model_choice = data.get('model_choice', 'auto')
        
        if not user_message.strip():
            return jsonify({'error': 'Please enter a valid question'}), 400
        
        logger.info(f"Processing message from user {user_id}: {user_message[:50]}... (model: {model_choice})")
        
        # Initialize chat history for this user if needed
        if user_id not in chat_histories:
            chat_histories[user_id] = []
        
        # Get the active connection type
        db_connection = session.get('db_connection', {})
        db_type = db_connection.get('db_type', 'graphdb')  # Default to GraphDB
        
        # Get response from LangGraph chatbot with the specified DB type and model choice
        response, updated_history = get_response(
            user_message, 
            chat_histories[user_id], 
            db_type=db_type,
            connection_config=db_connection,
            model_choice=model_choice
        )
        chat_histories[user_id] = updated_history
        
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/connection-types', methods=['GET'])
def connection_types():
    """Get available database connection types."""
    try:
        available_types = get_connection_types()
        return jsonify(available_types)
    except Exception as e:
        logger.error(f"Error retrieving connection types: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/set-connection', methods=['POST'])
def set_connection():
    """Set the database connection configuration."""
    try:
        data = request.get_json()
        db_type = data.get('db_type', 'graphdb')
        
        # Save connection details to session
        session['db_connection'] = data
        
        # Configure the connection in the backend
        success, error = set_connection_config(db_type, data)
        
        if success:
            return jsonify({'success': True, 'message': f'Successfully connected to {db_type}'})
        else:
            return jsonify({'success': False, 'error': error})
    except Exception as e:
        logger.error(f"Error setting connection: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/clear-connection', methods=['POST'])
def clear_connection():
    """Clear the current database connection."""
    try:
        if 'db_connection' in session:
            del session['db_connection']
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error clearing connection: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)