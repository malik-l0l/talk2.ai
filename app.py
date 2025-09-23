from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import os
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import uuid

from config import *

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Configuration
GROQ_API_KEY = GROQ_API_KEY # Replace with your actual API key

# Initialize services
client = MongoClient(MONGODB_URI)
db = client.personality_simulator
users_collection = db.users
chats_collection = db.chats

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_collection(personality_name):
    """Get or create a ChromaDB collection for a personality"""
    try:
        collection = chroma_client.get_collection(name=personality_name)
    except:
        collection = chroma_client.create_collection(name=personality_name)
    return collection

def search_relevant_chunks(query, personality, top_k=3):
    """Search for relevant chunks in the vector database"""
    try:
        collection = get_collection(personality)
        
        # Create embedding for the query
        query_embedding = embedding_model.encode([query])
        
        # Search for similar chunks
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        if results['documents'] and results['documents'][0]:
            return results['documents'][0]
        return []
    except Exception as e:
        print(f"Error searching chunks: {e}")
        return []

def generate_response(query, context_chunks, personality):
    """Generate response using Groq API with context"""
    try:
        # Combine context chunks
        context = "\n".join(context_chunks)
        
        # Create personality-specific prompt
        personality_prompts = {
            "elonmusk": f"""You are Elon Musk. Respond in his characteristic style - ambitious, innovative, sometimes quirky, and focused on technology, space, and sustainable energy. Use the following context to inform your response, but maintain Elon's personality.

Context: {context}

User: {query}

Elon Musk:""",
            "gandhi": f"""You are Mahatma Gandhi. Respond with wisdom, non-violence, and spiritual insight. Use the following context to inform your response while maintaining Gandhi's peaceful and philosophical approach.

Context: {context}

User: {query}

Gandhi:"""
        }

        prompt = personality_prompts.get(personality, f"""You are {personality}. Use the following context to respond as this personality would.

Context: {context}

User: {query}

{personality}:""")

        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile" ,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm having trouble responding right now. Please try again."

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = users_collection.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if users_collection.find_one({'username': username}):
            flash('Username already exists')
        else:
            hashed_password = generate_password_hash(password)
            users_collection.insert_one({
                'username': username,
                'password': hashed_password,
                'created_at': datetime.now()
            })
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/chat/<personality>')
def chat(personality):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get chat history for this user and personality
    chat_history = list(chats_collection.find({
        'user_id': session['user_id'],
        'personality': personality
    }).sort('timestamp', 1))
    
    return render_template('chat.html', personality=personality, chat_history=chat_history)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    query = data.get('message', '')
    personality = data.get('personality', '')
    
    if not query or not personality:
        return jsonify({'error': 'Missing message or personality'}), 400
    
    try:
        # Search for relevant chunks
        context_chunks = search_relevant_chunks(query, personality)
        
        # Generate response
        response = generate_response(query, context_chunks, personality)
        
        # Save chat to database
        chat_entry = {
            'user_id': session['user_id'],
            'personality': personality,
            'user_message': query,
            'bot_response': response,
            'timestamp': datetime.now(),
            'session_id': str(uuid.uuid4())
        }
        chats_collection.insert_one(chat_entry)
        
        return jsonify({
            'response': response,
            'success': True
        })
    
    except Exception as e:
        print(f"Error in chat API: {e}")
        return jsonify({'error': 'Failed to generate response'}), 500

@app.route('/api/chat_history/<personality>')
def get_chat_history(personality):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        chats = list(chats_collection.find({
            'user_id': session['user_id'],
            'personality': personality
        }).sort('timestamp', 1).limit(50))
        
        # Convert ObjectId to string for JSON serialization
        for chat in chats:
            chat['_id'] = str(chat['_id'])
            chat['timestamp'] = chat['timestamp'].isoformat()
        
        return jsonify({'chats': chats})
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return jsonify({'error': 'Failed to fetch chat history'}), 500

if __name__ == '__main__':
    app.run(debug=True)