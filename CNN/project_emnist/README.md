# EMNIST Alphanumeric Recognizer

Handwritten character recognition using CNN on EMNIST dataset.

## Features
- ✅ Recognizes digits (0-9)
- ✅ Recognizes uppercase letters (A-Z)
- ✅ Recognizes lowercase letters (a-z)
- ✅ Interactive web drawing canvas
- ✅ Real-time predictions with confidence scores

## Model
- **Architecture**: LeNet (Conv2D → ReLU → MaxPool → Dense)
- **Dataset**: EMNIST Balanced (70,000 training samples)
- **Accuracy**: ~83%

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Start Server
```bash
python app.py
```

Visit: `http://localhost:5000`

## How It Works

1. Draw a character on canvas
2. Click "Predict"
3. Model returns top 3 predictions with confidence

## Files
- `app.py` - Flask server + preprocessing
- `train_emnist_lenet.py` - Model training script
- `emnist_lenet_model.h5` - Trained model
- `label_mapping.csv` - Character mappings (0-9, A-Z, a-z)

## Key Technical Details

**Preprocessing Pipeline:**
```
Canvas Image (280×280) 
  → Resize to 28×28
  → Grayscale convert
  → Invert (white bg → black bg)
  → Normalize (0-1)
  → Flip axis=1
  → Predict
```

**Why Flip?**
EMNIST training data required horizontal flip for correct orientation.

## Results
- Digits: 85%+ accuracy
- Uppercase: 82%+ accuracy  
- Lowercase: 81%+ accuracy

## Future Improvements
- Add batch prediction
- Improve model accuracy (ResNet)
- Deploy to cloud (Heroku/AWS)