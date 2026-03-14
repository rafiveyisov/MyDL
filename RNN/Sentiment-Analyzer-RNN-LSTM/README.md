# RNN vs. LSTM Sentiment Analyzer 🎬

A complete end-to-end deep learning web application that compares SimpleRNN and LSTM models for movie review sentiment analysis in real-time.

## 📋 Project Overview

This project demonstrates the difference between two recurrent neural network architectures:
- **SimpleRNN**: Basic recurrent model with simpler memory mechanism
- **LSTM**: Advanced model with long short-term memory cells

Users can type movie reviews and instantly see how both models perceive the sentiment, with side-by-side comparison of confidence scores.

## 🏗️ Architecture

### Part 1: Model Training (The Brain)
- **Model A (SimpleRNN)**: Embedding(10000, 128) → SimpleRNN(64) → Dense(1)
- **Model B (LSTM)**: Embedding(10000, 128) → LSTM(64) → Dense(1)
- Trained on IMDB dataset with 50,000 movie reviews
- Binary classification (Positive/Negative sentiment)

### Part 2: Backend (The Engine)
- Flask server with RESTful API
- Loads both models simultaneously
- `/predict` endpoint processes text and returns JSON results
- Real-time text preprocessing and tokenization

### Part 3: Frontend (The Interface)
- Modern, responsive web interface
- Side-by-side model comparison
- Real-time sentiment analysis
- Beautiful gradient design with animations

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended (for model training)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Models

This will train both models on the IMDB dataset and save them:

```bash
python train_models.py
```

**Training time**: ~15-30 minutes depending on your hardware

**Output files**:
- `simple_rnn_model.h5` - Trained SimpleRNN model
- `lstm_model.h5` - Trained LSTM model  
- `word_index.pkl` - Word vocabulary for tokenization

### Step 3: Start the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### Step 4: Open the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

## 💻 Usage

1. **Enter a Review**: Type or paste a movie review in the text area
2. **Analyze**: Click the "Analyze Sentiment" button
3. **View Results**: See side-by-side predictions from both models
   - Confidence scores (0-100%)
   - Sentiment labels (Positive/Negative/Neutral)
   - Visual progress bars

### Example Reviews to Try

**Positive Review**:
```
This movie was absolutely amazing! The acting was superb, 
the plot was engaging, and the cinematography was beautiful. 
I highly recommend it to everyone.
```

**Negative Review**:
```
Terrible waste of time. The plot made no sense, the acting 
was wooden, and I regretted every minute I spent watching this.
```

**Mixed Review**:
```
The movie had some good moments and decent special effects, 
but the story was predictable and the ending was disappointing.
```

## 📊 API Documentation

### Endpoint: `/predict`

**Method**: POST

**Request Body**:
```json
{
  "text": "Your movie review text here"
}
```

**Response**:
```json
{
  "rnn_result": {
    "score": 0.6523,
    "percentage": 65.23,
    "label": "Positive"
  },
  "lstm_result": {
    "score": 0.9187,
    "percentage": 91.87,
    "label": "Positive"
  },
  "text_length": 45
}
```

### Endpoint: `/health`

**Method**: GET

**Response**:
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

## 🎯 Model Performance

Typical accuracy on IMDB test set:
- **SimpleRNN**: ~83-85% accuracy
- **LSTM**: ~86-88% accuracy

LSTM generally performs better due to its ability to capture long-term dependencies in text.

## 📁 Project Structure

```
rnn-lstm-sentiment-analyzer/
├── train_models.py          # Model training script
├── app.py                   # Flask backend server
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── templates/
│   └── index.html          # Frontend interface
├── simple_rnn_model.h5     # Trained SimpleRNN (generated)
├── lstm_model.h5           # Trained LSTM (generated)
└── word_index.pkl          # Word vocabulary (generated)
```

## 🔧 Configuration

You can modify these parameters in `train_models.py`:

- `MAX_FEATURES`: Vocabulary size (default: 10,000)
- `MAX_LEN`: Maximum review length (default: 200 words)
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Number of training epochs (default: 10)

## 🎨 Features

- ✅ Real-time sentiment analysis
- ✅ Side-by-side model comparison
- ✅ Beautiful, modern UI with animations
- ✅ Confidence score visualization
- ✅ Responsive design (mobile-friendly)
- ✅ Error handling and validation
- ✅ Progress indicators
- ✅ Clear/reset functionality

## 🐛 Troubleshooting

**Issue**: Models not loading
- **Solution**: Make sure you've run `train_models.py` first

**Issue**: Port 5000 already in use
- **Solution**: Change the port in `app.py` (last line)

**Issue**: Out of memory during training
- **Solution**: Reduce `BATCH_SIZE` in `train_models.py`

**Issue**: Slow predictions
- **Solution**: This is normal for CPU inference. For faster predictions, use a GPU-enabled environment

## 📚 Technical Details

### Text Preprocessing
1. Convert to lowercase
2. Remove special characters
3. Split into words
4. Convert words to integer indices using IMDB word index
5. Pad sequences to fixed length (200)

### Model Architecture

**SimpleRNN**:
- Embedding layer: 10,000 vocabulary → 128 dimensions
- SimpleRNN layer: 64 units with 20% dropout
- Dense output: Sigmoid activation for binary classification

**LSTM**:
- Embedding layer: 10,000 vocabulary → 128 dimensions  
- LSTM layer: 64 units with 20% dropout
- Dense output: Sigmoid activation for binary classification

### Training
- Loss function: Binary crossentropy
- Optimizer: Adam
- Early stopping with patience=2
- 20% validation split

## 🤝 Contributing

Feel free to fork this project and make improvements!

## 📄 License

MIT License - feel free to use this project for learning and development.

## 🎓 Learning Outcomes

By completing this project, you will understand:
- How to train RNN and LSTM models for NLP tasks
- Building REST APIs with Flask
- Integrating machine learning models into web applications
- Real-time prediction systems
- Frontend-backend communication
- Text preprocessing for deep learning

## 📞 Support

If you encounter any issues or have questions, please create an issue in the repository.

---

**Built with ❤️ for learning Deep Learning and Full-Stack Development**
