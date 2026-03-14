"""
COMPLETE WORKING VERSION
Everything in one file - no separate templates folder needed
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import pickle
import re

print("\n" + "="*60)
print("Loading RNN vs LSTM Sentiment Analyzer")
print("="*60)

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Configuration
MAX_LEN = 200
MAX_FEATURES = 10000

# Load models
print("Loading models...")
try:
    model_rnn = load_model('simple_rnn_model.h5')
    model_lstm = load_model('lstm_model.h5')
    print("✓ Models loaded")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    exit(1)

# Load word index
print("Loading word index...")
try:
    with open('word_index.pkl', 'rb') as f:
        word_index = pickle.load(f)
    print(f"✓ Word index loaded ({len(word_index)} words)")
except Exception as e:
    print(f"❌ Failed to load word index: {e}")
    exit(1)

print("="*60 + "\n")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    sequence_data = []
    
    for word in words:
        if word in word_index:
            # Most IMDB models use offset 3 for special tokens
            idx = word_index[word] + 3
            if idx < MAX_FEATURES:
                sequence_data.append(idx)
    
    if not sequence_data:
        sequence_data = [2] # ID for unknown words
    
    # Ensure sequence is padded correctly
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences([sequence_data], maxlen=MAX_LEN)

def interpret_score(score):
    """Convert score to label"""
    if score >= 0.6:
        return "Positive"
    elif score <= 0.4:
        return "Negative"
    else:
        return "Neutral"

# HTML content embedded in Python
HTML_CONTENT = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RNN vs LSTM Sentiment Analyzer</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@600;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Space Mono', monospace;
            background: #0a0e27;
            color: #e8eaed;
            min-height: 100vh;
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 {
            font-family: 'Syne', sans-serif;
            font-size: 3rem;
            background: linear-gradient(135deg, #00d4ff, #ff006e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
        }
        .input-section {
            background: #151b3d;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        textarea {
            width: 100%;
            min-height: 150px;
            background: #0a0e27;
            border: 2px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1rem;
            font-family: 'Space Mono', monospace;
            font-size: 1rem;
            color: #e8eaed;
            resize: vertical;
        }
        textarea:focus {
            outline: none;
            border-color: #00d4ff;
        }
        button {
            background: linear-gradient(135deg, #00d4ff, #ff006e);
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 12px;
            font-family: 'Syne', sans-serif;
            font-size: 1rem;
            cursor: pointer;
            margin-top: 1rem;
            width: 100%;
        }
        button:hover { transform: translateY(-2px); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .results {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-top: 2rem;
        }
        .card {
            background: #151b3d;
            border: 2px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
        }
        .card.rnn { border-top: 4px solid #00d4ff; }
        .card.lstm { border-top: 4px solid #ff006e; }
        .model-name {
            font-family: 'Syne', sans-serif;
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }
        .rnn .model-name { color: #00d4ff; }
        .lstm .model-name { color: #ff006e; }
        .score {
            font-size: 3rem;
            font-weight: bold;
            margin: 1rem 0;
        }
        .label {
            font-size: 1.5rem;
            margin: 1rem 0;
        }
        .label.positive { color: #10b981; }
        .label.negative { color: #ef4444; }
        .label.neutral { color: #f59e0b; }
        .error {
            background: rgba(239,68,68,0.1);
            border: 1px solid #ef4444;
            color: #ef4444;
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            display: none;
        }
        .error.show { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>RNN vs LSTM</h1>
        
        <div class="error" id="error"></div>
        
        <div class="input-section">
            <textarea id="text" placeholder="Type your movie review here..."></textarea>
            <button onclick="analyze()" id="btn">Analyze Sentiment</button>
        </div>
        
        <div class="results" id="results" style="display: none;">
            <div class="card rnn">
                <div class="model-name">SimpleRNN</div>
                <div class="score" id="rnn-score">--</div>
                <div class="label" id="rnn-label">--</div>
            </div>
            <div class="card lstm">
                <div class="model-name">LSTM</div>
                <div class="score" id="lstm-score">--</div>
                <div class="label" id="lstm-label">--</div>
            </div>
        </div>
    </div>
    
    <script>
        async function analyze() {
            const text = document.getElementById('text').value.trim();
            const btn = document.getElementById('btn');
            const error = document.getElementById('error');
            const results = document.getElementById('results');
            
            error.classList.remove('show');
            
            if (!text) {
                error.textContent = 'Please enter some text';
                error.classList.add('show');
                return;
            }
            
            btn.disabled = true;
            btn.textContent = 'Analyzing...';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                
                const data = await response.json();
                
                if (!response.ok || data.error) {
                    throw new Error(data.error || 'Prediction failed');
                }
                
                // Update RNN
                document.getElementById('rnn-score').textContent = data.rnn_result.percentage + '%';
                document.getElementById('rnn-label').textContent = data.rnn_result.label;
                document.getElementById('rnn-label').className = 'label ' + data.rnn_result.label.toLowerCase();
                
                // Update LSTM
                document.getElementById('lstm-score').textContent = data.lstm_result.percentage + '%';
                document.getElementById('lstm-label').textContent = data.lstm_result.label;
                document.getElementById('lstm-label').className = 'label ' + data.lstm_result.label.toLowerCase();
                
                results.style.display = 'grid';
                
            } catch (err) {
                error.textContent = 'Error: ' + err.message;
                error.classList.add('show');
            } finally {
                btn.disabled = false;
                btn.textContent = 'Analyze Sentiment';
            }
        }
    </script>
</body>
</html>'''

@app.route('/')
def home():
    """Serve HTML page"""
    return HTML_CONTENT

@app.route('/predict', methods=['POST'])
def predict():
    print("\n" + "="*60)
    print("Request received!")
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data.get('text', '')
        print(f"Input Text: {text[:50]}...")

        # 1. Preprocess
        processed = preprocess_text(text)
        
        # 2. Predict
        # We use .tolist() or float() to ensure the data is JSON-serializable
        rnn_pred = model_rnn.predict(processed, verbose=0)
        lstm_pred = model_lstm.predict(processed, verbose=0)
        
        rnn_score = float(rnn_pred[0][0])
        lstm_score = float(lstm_pred[0][0])

        print(f"Scores -> RNN: {rnn_score:.4f}, LSTM: {lstm_score:.4f}")

        response = {
            'rnn_result': {
                'score': round(rnn_score, 4),
                'percentage': round(rnn_score * 100, 2),
                'label': interpret_score(rnn_score)
            },
            'lstm_result': {
                'score': round(lstm_score, 4),
                'percentage': round(lstm_score * 100, 2),
                'label': interpret_score(lstm_score)
            }
        }
        return jsonify(response)

    except Exception as e:
        print("❌ PREDICTION ERROR:")
        import traceback
        traceback.print_exc() # This prints the EXACT error to your console
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("🚀 Server starting on http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)