from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
import logging
import os
import sys
import time
import json

# Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "methods": "*"}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    return response

def find_file(filename):
    paths = [os.getcwd(), os.path.dirname(os.path.abspath(__file__)),
             os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    for p in paths:
        f = os.path.join(p, filename)
        if os.path.exists(f): return f
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for root, dirs, files in os.walk(base):
        if filename in files: return os.path.join(root, filename)
    return None

# Model
print("="*60)
print("🚀 EMNIST Server")
print("="*60)

model = None
model_path = find_file('emnist_lenet_model.h5')
if model_path:
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded: {model.input_shape}")
    except Exception as e:
        print(f"❌ Model error: {e}")
else:
    print("❌ Model not found")

# Labels - DEBUG VERSION
label_map = {}
try:
    f = find_file('label_mapping.csv')
    if f:
        df = pd.read_csv(f)
        # Ensure correct types
        label_map = {int(row['label']): str(row['character']) for _, row in df.iterrows()}
        print(f"✅ Mapping loaded: {len(label_map)} classes")
        print(f"   Sample: {dict(list(label_map.items())[:5])}")
except Exception as e:
    print(f"⚠️ Mapping error: {e}")

# Fallback mapping
if not label_map:
    print("📝 Using default mapping")
    for i in range(10): label_map[i] = str(i)
    for i in range(26): label_map[10+i] = chr(ord('A')+i)
    for i in range(26): label_map[36+i] = chr(ord('a')+i)

print(f"   Final mapping count: {len(label_map)}")
print(f"   Keys sample: {list(label_map.keys())[:10]}")
print(f"   Values sample: {list(label_map.values())[:10]}")

# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_emnist(image_data):
    try:
        _, encoded = image_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_bytes)).convert('L')
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        
        arr = 255.0 - arr
        arr = arr / 255.0
        arr = arr.reshape(1, 28, 28, 1)
        
        # ✅ VARIANT 1: FLIP ÖNCE
        arr = arr.reshape(1, 28, 28, 1)
        arr = np.flip(arr, axis=1)
        
        return arr
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/test')
def test():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': len(label_map),
        'label_sample': dict(list(label_map.items())[:5])
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        logger.info(f"Received request with image length: {len(data.get('image', ''))}")
        
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data'}), 400
        
        # Process
        processed = preprocess_emnist(data['image'])
        if processed is None:
            return jsonify({'success': False, 'error': 'Image processing failed'}), 400
        
        logger.info(f"Processed shape: {processed.shape}, range: [{processed.min():.3f}, {processed.max():.3f}]")
        
        # Predict
        preds = model.predict(processed, verbose=0)[0]
        
        # Get top 3
        top_indices = np.argsort(preds)[-3:][::-1]
        
        logger.info(f"Raw predictions top 3 indices: {top_indices}")
        logger.info(f"Raw confidences: {[float(preds[i]) for i in top_indices]}")
        
        # Build response with DEBUG info
        predictions_list = []
        for idx in top_indices:
            label_idx = int(idx)
            char = label_map.get(label_idx, f"UNKNOWN({label_idx})")
            conf = float(preds[idx])
            predictions_list.append({
                'character': char,
                'confidence': conf,
                'label_index': label_idx
            })
            logger.info(f"  Index {label_idx} -> '{char}' ({conf:.4f})")
        
        # CRITICAL: Check if top prediction exists
        top_pred = predictions_list[0]
        
        response = {
            'success': True,
            'top_character': top_pred['character'],
            'top_confidence': top_pred['confidence'],
            'top_label_index': top_pred['label_index'],
            'predictions': predictions_list,
            'debug': {
                'input_shape': list(processed.shape),
                'pixel_min': float(processed.min()),
                'pixel_max': float(processed.max()),
                'pixel_mean': float(processed.mean()),
                'raw_top_indices': [int(i) for i in top_indices],
                'label_map_keys_sample': list(label_map.keys())[:10]
            }
        }
        
        logger.info(f"Response: {response['top_character']} ({response['top_confidence']:.2%})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        return jsonify({
            'success': False, 
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/raw_predict', methods=['POST'])
def raw_predict():
    """Debug endpoint - shows raw model output"""
    if model is None:
        return jsonify({'error': 'No model'}), 500
    
    try:
        data = request.get_json()
        processed = preprocess_emnist(data['image'])
        if processed is None:
            return jsonify({'error': 'Process failed'}), 400
        
        preds = model.predict(processed, verbose=0)[0]
        
        # Return ALL predictions sorted
        all_preds = [(int(i), float(preds[i]), label_map.get(i, '?')) for i in range(len(preds))]
        all_preds.sort(key=lambda x: x[1], reverse=True)
        
        return jsonify({
            'success': True,
            'top_10': [
                {'index': idx, 'char': char, 'confidence': conf}
                for idx, conf, char in all_preds[:10]
            ],
            'shape': list(processed.shape),
            'pixel_stats': {
                'min': float(processed.min()),
                'max': float(processed.max()),
                'mean': float(processed.mean())
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EMNIST Debug</title>
    <style>
        body { font-family: Arial; background: #1a1a2e; color: white; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: #16213e; padding: 30px; border-radius: 15px; }
        h1 { color: #e94560; text-align: center; }
        canvas { background: white; border: 3px solid #e94560; border-radius: 10px; cursor: crosshair; display: block; margin: 20px auto; }
        .buttons { text-align: center; margin: 20px 0; }
        button { background: #e94560; color: white; border: none; padding: 12px 25px; margin: 5px; border-radius: 8px; cursor: pointer; font-size: 1em; }
        button:hover { background: #ff6b6b; }
        .result { background: #0f3460; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .big { font-size: 4em; color: #16c79a; text-align: center; }
        .debug { background: #2c3e50; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 11px; overflow-x: auto; margin-top: 15px; }
        .error { color: #e74c3c; background: #fadbd8; padding: 15px; border-radius: 8px; margin: 10px 0; color: #c0392b; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { border: 1px solid #333; padding: 10px; text-align: center; }
        th { background: #0f3460; }
        tr:hover { background: #1a1a2e; }
        .highlight { background: #16c79a !important; color: #1a1a2e; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 EMNIST Debug Console</h1>
        
        <canvas id="canvas" width="280" height="280"></canvas>
        
        <div class="buttons">
            <button onclick="clearCanvas()">🗑️ Clear</button>
            <button onclick="predict()">🔍 Predict</button>
            <button onclick="rawPredict()" style="background: #3498db;">🔬 Raw Debug</button>
        </div>

        <div id="result" class="result" style="display: none;">
            <h3>Result:</h3>
            <div class="big" id="predChar">?</div>
            <div style="text-align: center; color: #aaa;" id="predConf"></div>
            <div class="debug" id="debugInfo"></div>
        </div>

        <div id="rawResult" style="display: none;">
            <h3>Top 10 Predictions:</h3>
            <table id="rawTable">
                <tr><th>Rank</th><th>Index</th><th>Character</th><th>Confidence</th></tr>
            </table>
            <div class="debug" id="rawDebug"></div>
        </div>

        <div class="debug" id="log" style="margin-top: 20px; max-height: 200px; overflow-y: auto;">
            <div>> System ready...</div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false, lastX = 0, lastY = 0;

        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = '#000000';
        clearCanvas();

        function getPos(e) {
            const rect = canvas.getBoundingClientRect();
            const clientX = e.clientX || (e.touches && e.touches[0].clientX);
            const clientY = e.clientY || (e.touches && e.touches[0].clientY);
            return {
                x: (clientX - rect.left) * (canvas.width / rect.width),
                y: (clientY - rect.top) * (canvas.height / rect.height)
            };
        }

        canvas.addEventListener('mousedown', e => {
            isDrawing = true;
            const pos = getPos(e);
            [lastX, lastY] = [pos.x, pos.y];
        });

        canvas.addEventListener('mousemove', e => {
            if (!isDrawing) return;
            const pos = getPos(e);
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
            [lastX, lastY] = [pos.x, pos.y];
        });

        canvas.addEventListener('mouseup', () => isDrawing = false);
        canvas.addEventListener('mouseout', () => isDrawing = false);

        canvas.addEventListener('touchstart', e => {
            e.preventDefault();
            const t = e.touches[0];
            canvas.dispatchEvent(new MouseEvent('mousedown', {clientX: t.clientX, clientY: t.clientY}));
        });
        canvas.addEventListener('touchmove', e => {
            e.preventDefault();
            const t = e.touches[0];
            canvas.dispatchEvent(new MouseEvent('mousemove', {clientX: t.clientX, clientY: t.clientY}));
        });
        canvas.addEventListener('touchend', e => {
            e.preventDefault();
            canvas.dispatchEvent(new MouseEvent('mouseup', {}));
        });

        function clearCanvas() {
            ctx.fillStyle = '#FFFFFF';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('result').style.display = 'none';
            document.getElementById('rawResult').style.display = 'none';
            log('Canvas cleared');
        }

        function log(msg) {
            const div = document.getElementById('log');
            const time = new Date().toLocaleTimeString();
            div.innerHTML += `<div>[${time}] ${msg}</div>`;
            div.scrollTop = div.scrollHeight;
            console.log(msg);
        }

        async function predict() {
            const imageData = canvas.toDataURL('image/png');
            log('Sending prediction...');
            
            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image: imageData})
                });
                
                const data = await res.json();
                console.log('Full response:', data);
                
                if (data.success) {
                    document.getElementById('predChar').textContent = data.top_character || '?';
                    document.getElementById('predConf').textContent = 
                        `Confidence: ${(data.top_confidence * 100).toFixed(1)}% (Index: ${data.top_label_index})`;
                    document.getElementById('debugInfo').innerHTML = 
                        '<pre>' + JSON.stringify(data.debug, null, 2) + '</pre>';
                    document.getElementById('result').style.display = 'block';
                    
                    log(`Result: "${data.top_character}" (${(data.top_confidence*100).toFixed(1)}%)`);
                    
                    // Check for issues
                    if (!data.top_character || data.top_character.includes('UNKNOWN')) {
                        log('⚠️ WARNING: Character mapping issue detected!');
                    }
                } else {
                    log('❌ Error: ' + data.error);
                    if (data.traceback) {
                        console.error(data.traceback);
                    }
                }
            } catch (e) {
                log('❌ Failed: ' + e.message);
                console.error(e);
            }
        }

        async function rawPredict() {
            const imageData = canvas.toDataURL('image/png');
            log('Getting raw predictions...');
            
            try {
                const res = await fetch('/raw_predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image: imageData})
                });
                
                const data = await res.json();
                console.log('Raw data:', data);
                
                if (data.success) {
                    const table = document.getElementById('rawTable');
                    // Clear old rows
                    while (table.rows.length > 1) table.deleteRow(1);
                    
                    data.top_10.forEach((item, idx) => {
                        const row = table.insertRow();
                        if (idx === 0) row.className = 'highlight';
                        row.innerHTML = `
                            <td>${idx + 1}</td>
                            <td>${item.index}</td>
                            <td style="font-size: 1.5em;"><b>${item.char}</b></td>
                            <td>${(item.confidence * 100).toFixed(2)}%</td>
                        `;
                    });
                    
                    document.getElementById('rawDebug').innerHTML = 
                        '<pre>' + JSON.stringify(data.pixel_stats, null, 2) + '</pre>';
                    document.getElementById('rawResult').style.display = 'block';
                    
                    log('Raw predictions loaded - check table above');
                } else {
                    log('Raw predict error: ' + data.error);
                }
            } catch (e) {
                log('Raw predict failed: ' + e.message);
            }
        }

        log('Ready! Draw a character and click Predict or Raw Debug');
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    print(f"\n🌐 Server: http://localhost:5000")
    print(f"📊 Model: {'✅' if model else '❌'}")
    print(f"🏷️  Classes: {len(label_map)}")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)