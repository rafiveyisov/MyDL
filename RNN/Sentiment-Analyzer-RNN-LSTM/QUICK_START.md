# 🚀 QUICK START GUIDE

## Setup in 3 Steps

### 1️⃣ Install Dependencies (1 minute)
```bash
pip install -r requirements.txt
```

### 2️⃣ Train Models (15-30 minutes)
```bash
python train_models.py
```

This creates:
- ✅ simple_rnn_model.h5
- ✅ lstm_model.h5  
- ✅ word_index.pkl

### 3️⃣ Start Server (instant)
```bash
python app.py
```

Open: http://localhost:5000

---

## 📦 What You Get

### Part 1: The Brain (Models)
- **SimpleRNN Model**: Basic recurrent architecture
- **LSTM Model**: Advanced long short-term memory
- Trained on 50,000 IMDB movie reviews

### Part 2: The Engine (Backend)
- Flask REST API
- `/predict` endpoint for real-time analysis
- Loads both models simultaneously
- JSON response with confidence scores

### Part 3: The Interface (Frontend)
- Modern, responsive web design
- Side-by-side model comparison
- Real-time sentiment visualization
- Beautiful gradient animations

---

## 🎯 All Requirements Met

✅ **Objective**: Compare RNN vs LSTM in real-time web app
✅ **Part 1**: Both models trained and saved as .h5 files
✅ **Part 2**: Flask backend with /predict endpoint
✅ **Part 3**: HTML interface with side-by-side cards

---

## 💡 Test Examples

**Positive Review**:
```
This movie was absolutely amazing! The acting was superb 
and the plot kept me engaged from start to finish.
```

**Negative Review**:
```
Terrible waste of time. The plot made no sense and 
the acting was awful. I want my money back.
```

**Neutral Review**:
```
The movie was okay. Some parts were good but 
overall it was just average entertainment.
```

---

## 🔍 Understanding the Results

- **Score**: 0.0-1.0 (converted to 0-100%)
- **Label**: 
  - Positive (≥60%)
  - Negative (≤40%)
  - Neutral (40-60%)

LSTM typically shows:
- Higher confidence on clear sentiments
- Better handling of complex sentences
- More accurate predictions overall

---

## 📊 Expected Performance

- SimpleRNN: ~83-85% accuracy
- LSTM: ~86-88% accuracy
- Training time: 15-30 minutes
- Prediction: < 1 second

---

## 🛠️ Optional: Quick Test

Before training, verify your setup:
```bash
python test_setup.py
```

This checks:
- Dependencies installed correctly
- Model architectures valid
- Prediction pipeline working

---

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `train_models.py` | Trains both models on IMDB dataset |
| `app.py` | Flask server with prediction endpoint |
| `templates/index.html` | Frontend interface |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |
| `test_setup.py` | System verification |
| `start.sh` | One-click setup script |

---

## ⚡ Alternative: One-Click Setup

Linux/Mac users can run:
```bash
./start.sh
```

This automatically:
1. Installs dependencies
2. Trains models
3. Starts the server

---

## 🎨 Design Features

- Cyberpunk-inspired gradient theme
- Smooth animations and transitions
- Responsive mobile-friendly layout
- Real-time loading indicators
- Error handling and validation
- Progress bars for visual feedback

---

## 🤝 Need Help?

1. Check the full README.md for detailed docs
2. Run test_setup.py to verify installation
3. Make sure all dependencies are installed
4. Ensure port 5000 is available

**Happy Analyzing! 🎬**
