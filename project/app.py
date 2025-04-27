

from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import os
import functools
from datetime import datetime
from uuid import uuid4
import firebase_admin
from firebase_admin import credentials, db, auth as firebase_auth
import pyrebase
import PyPDF2
import docx
import re
from neo4j import GraphDatabase
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json









class GraphEmbedder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def extract_graph_features(self):
        """Extract features for each disease based on its connections"""
        with self.driver.session() as session:
            # Get all diseases with their connected entities
            result = session.run("""
            MATCH (d:Disease)
            OPTIONAL MATCH (d)<-[:INDICATES]-(s:Symptom)
            OPTIONAL MATCH (d)-[:TREATED_BY]->(sp:Specialist)
            OPTIONAL MATCH (d)-[:AFFECTS]->(bp:BodyPart)
            RETURN d.name as disease, 
                   id(d) as node_id,
                   count(DISTINCT s) as symptom_count,
                   count(DISTINCT sp) as specialist_count,
                   count(DISTINCT bp) as bodypart_count,
                   labels(d)[0] as label
            """)
            
            features = [dict(record) for record in result]
            return pd.DataFrame(features)
    
    def generate_embeddings(self, n_components=2):
        """Generate embeddings using PCA on the extracted features"""
        features_df = self.extract_graph_features()
        
        # Select and scale numerical features
        numerical_cols = ['symptom_count', 'specialist_count', 'bodypart_count']
        X = features_df[numerical_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA - using min(n_samples, n_features) components
        n_components = min(n_components, X_scaled.shape[1])
        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(X_scaled)
        
        # Create embedding dictionary
        embedding_dict = {}
        for idx, row in features_df.iterrows():
            embedding_dict[row['node_id']] = embeddings[idx]
            embedding_dict[row['disease']] = embeddings[idx]
        
        # Save additional information for reference
        embedding_info = {
            'embeddings': embedding_dict,
            'feature_means': scaler.mean_,
            'feature_stds': scaler.scale_,
            'pca_components': pca.components_,
            'disease_names': features_df['disease'].tolist()
        }
        
        return embedding_info

# Usage
try:
    embedder = GraphEmbedder("bolt://44.201.21.92", "neo4j", "partitions-slave-diagrams")
    embedding_info = embedder.generate_embeddings(n_components=2)
    
    # Save embeddings
    with open('disease_embeddings.pkl', 'wb') as f:
        pickle.dump(embedding_info, f)
    
    print("Embeddings generated successfully!")
    print(f"Number of diseases processed: {len(embedding_info['disease_names'])}")
    print(f"Example embedding for first disease: {embedding_info['embeddings'][embedding_info['disease_names'][0]]}")
    
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    embedder.close()



# ---------------------------- Setup ----------------------------
app = Flask(__name__)
app.secret_key = 'your-very-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'docx', 'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# -------------------------- Firebase Admin ----------------------

cred = credentials.Certificate('/etc/secrets/icliniq-21dd7-firebase-adminsdk-fbsvc-90d94153c6.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://icliniq-21dd7-default-rtdb.firebaseio.com'
})
chat_history_ref = db.reference('/chat_history')
user_ref = db.reference('/users')

# -------------------------- Pyrebase Client ----------------------
firebaseConfig = {
    "apiKey": "AIzaSyB6QSK2d6CYORAeiOccO3-gbGukgR5pBcc",
    "authDomain": "icliniq-21dd7.firebaseapp.com",
    "databaseURL": "https://icliniq-21dd7-default-rtdb.firebaseio.com",
    "projectId": "icliniq-21dd7",
    "storageBucket": "icliniq-21dd7.appspot.com",   # small fix here (you had a typo in .app)
    "messagingSenderId": "529610089080",
    "appId": "1:529610089080:web:b84c8ef107ae0012b8c38d"
}

firebase = pyrebase.initialize_app(firebaseConfig)
pyre_auth = firebase.auth()

# -------------------------- Helper Functions ----------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(filepath):
    try:
        if filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        elif filepath.endswith('.docx'):
            doc = docx.Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        elif filepath.endswith('.pdf'):
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join([page.extract_text() for page in reader.pages])
        return ""
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def save_chat_history(user_id, chat_id, chat_data):
    try:
        chat_history_ref.child(user_id).child(chat_id).set(chat_data)
        return True
    except Exception as e:
        print(f"Error saving chat: {e}")
        return False

def get_user_chat_history(user_id):
    try:
        return chat_history_ref.child(user_id).get() or {}
    except Exception as e:
        print(f"Error getting chat: {e}")
        return {}

def clear_chat_history(user_id):
    try:
        chat_history_ref.child(user_id).delete()
        return True
    except Exception as e:
        print(f"Error clearing chat: {e}")
        return False

def create_user(email, password, name):
    try:
        user = firebase_auth.create_user(
            email=email,
            password=password,
            display_name=name
        )
        user_ref.child(user.uid).set({
            'email': email,
            'name': name,
            'created_at': datetime.now().isoformat()
        })
        return user
    except Exception as e:
        print(f"Create user error: {e}")
        return None

def verify_user(email, password):
    try:
        user = pyre_auth.sign_in_with_email_and_password(email, password)
        info = pyre_auth.get_account_info(user['idToken'])
        user_info = info['users'][0]
        return {
            'uid': user_info['localId'],
            'email': user_info['email'],
            'name': user_info.get('displayName', '')
        }
    except Exception as e:
        print(f"Login error: {e}")
        return None

# -------------------------- Medical System Class ----------------------
class MedicalSystem:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._load_data()
        
    def _load_data(self):
        with self.driver.session() as session:
            result = session.run("MATCH (s:Symptom) RETURN s.name as name")
            self.all_symptoms = [record['name'] for record in result]
            self.symptom_lower_map = {s.lower(): s for s in self.all_symptoms}
            
            result = session.run("""
                MATCH (d:Disease)<-[:INDICATES]-(s:Symptom)
                RETURN d.name as disease, collect(s.name) as symptoms
            """)
            self.disease_symptoms = {record['disease']: record['symptoms'] for record in result}
            
            result = session.run("""
                MATCH (d:Disease)-[:TREATED_BY]->(s:Specialist)
                RETURN d.name as disease, collect(s.name) as specialists
            """)
            self.disease_specialists = {record['disease']: record['specialists'] for record in result}
        
        self.vectorizer = TfidfVectorizer(tokenizer=self._tokenize_medical)
        symptom_texts = [" ".join(symptoms) for symptoms in self.disease_symptoms.values()]
        self.disease_names = list(self.disease_symptoms.keys())
        self.symptom_vectors = self.vectorizer.fit_transform(symptom_texts)

    def _tokenize_medical(self, text):
        return re.findall(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*", text.lower())

    def analyze(self, text):
        input_clean = text.lower()
        symptoms_with_scores = []

        for symptom_lower, symptom in self.symptom_lower_map.items():
            if symptom_lower in input_clean:
                symptoms_with_scores.append((symptom, 100.0))
        
        for symptom in self.all_symptoms:
            if symptom in [s for s, _ in symptoms_with_scores]:
                continue
            matches = process.extract(symptom.lower(), [input_clean], limit=1)
            if matches and matches[0][1] > 90:
                symptoms_with_scores.append((symptom, matches[0][1]))
        
        input_terms = " ".join(self._tokenize_medical(input_clean))
        input_vec = self.vectorizer.transform([input_terms])
        cosine_scores = cosine_similarity(input_vec, self.symptom_vectors)[0]
        for idx, score in enumerate(cosine_scores):
            if score > 0.95:
                disease = self.disease_names[idx]
                for symptom in self.disease_symptoms[disease]:
                    if symptom not in [s for s, _ in symptoms_with_scores]:
                        symptoms_with_scores.append((symptom, score * 100))

        symptom_names = [s for s, _ in symptoms_with_scores]
        input_vec = self.vectorizer.transform([" ".join(symptom_names)])
        similarities = cosine_similarity(input_vec, self.symptom_vectors)[0]
        top_indices = np.argsort(similarities)[-5:][::-1]

        recommendations = []
        for idx in top_indices:
            disease = self.disease_names[idx]
            recommendations.append({
                'disease': disease,
                'confidence': float(similarities[idx]),
                'matching_symptoms': [s for s in symptom_names if s in self.disease_symptoms[disease]],
                'specialists': self.disease_specialists.get(disease, ["General Practitioner"])
            })
        
        return symptoms_with_scores, recommendations

    def close(self):
        self.driver.close()

# -------------------------- App Routes ----------------------

# Initialize Medical System
medical_system = MedicalSystem("bolt://44.201.21.92", "neo4j", "partitions-slave-diagrams")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = verify_user(email, password)
        if user:
            session['user'] = user
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            error = 'Invalid credentials.'
    return render_template('login.html', error=error)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'user' in session:
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            error = 'Passwords do not match.'
        else:
            user = create_user(email, password, name)
            if user:
                flash('Account created successfully! Please login.', 'success')
                return redirect(url_for('login'))
            else:
                error = 'Unable to create account.'
    return render_template('signup.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    symptoms = []
    results = []
    input_text = ""
    file_info = None

    if request.method == 'POST':
        if 'switch_chat' in request.form:
            session['current_chat_id'] = request.form['switch_chat']
        else:
            file = request.files.get('file')
            text_input = request.form.get('symptoms', '').strip()
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(filepath)
                input_text = extract_text_from_file(filepath)
                file_info = {
                    'name': filename,
                    'type': filename.rsplit('.', 1)[1].lower(),
                    'size': os.path.getsize(filepath)
                }
                os.remove(filepath)
            elif text_input:
                input_text = text_input

            if input_text:
                symptoms, results = medical_system.analyze(input_text)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                chat_data = {
                    'input': input_text,
                    'time': current_time,
                    'has_file': bool(file_info),
                    'file_info': file_info,
                    'symptoms': symptoms,
                    'results': results,
                    'last_updated': datetime.now().isoformat()
                }
                if not session.get('current_chat_id'):
                    session['current_chat_id'] = str(uuid4())
                save_chat_history(session['user']['uid'], session['current_chat_id'], chat_data)

    user_id = session['user']['uid']
    chat_history = get_user_chat_history(user_id)
    chat_history_list = []
    if chat_history:
        for chat_id, chat_data in chat_history.items():
            chat_data['id'] = chat_id
            chat_history_list.append(chat_data)
        chat_history_list.sort(key=lambda x: x.get('last_updated', ''), reverse=True)
    if not session.get('current_chat_id') and chat_history_list:
        session['current_chat_id'] = chat_history_list[0]['id']
    
    if session.get('current_chat_id') and chat_history:
        current_chat = chat_history.get(session['current_chat_id'])
        if current_chat:
            input_text = current_chat.get('input', '')
            file_info = current_chat.get('file_info')
            symptoms = current_chat.get('symptoms', [])
            results = current_chat.get('results', [])

    return render_template('index.html', symptoms=symptoms, results=results, input_text=input_text, file_info=file_info, chat_history=chat_history_list, current_chat_id=session.get('current_chat_id'), user=session.get('user'))

@app.route('/new-chat', methods=['POST'])
@login_required
def new_chat():
    session['current_chat_id'] = str(uuid4())
    flash('Started a new chat session.', 'success')
    return redirect(url_for('index'))

@app.route('/clear-history', methods=['POST'])
@login_required
def clear_history_route():
    clear_chat_history(session['user']['uid'])
    session.pop('current_chat_id', None)
    flash('Chat history cleared.', 'success')
    return redirect(url_for('index'))
@app.route('/google-login', methods=['POST'])
def google_login():
    try:
        id_token = request.json.get('idToken')
        decoded_token = firebase_auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        name = decoded_token.get('name', 'User')
        email = decoded_token.get('email')

        session['user'] = {
            'uid': uid,
            'email': email,
            'name': name
        }
        return {'success': True}, 200
    except Exception as e:
        print(f"Google login error: {e}")
        return {'success': False}, 400


@app.teardown_appcontext
def shutdown_session(exception=None):
    medical_system.close()

# ------------------------- Run App ------------------------
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=8080, debug=True)
