"""
Test Script - Quick verification of the sentiment analyzer
Tests the prediction pipeline without requiring trained models
"""

import numpy as np
from tensorflow.keras.preprocessing import sequence

# Configuration
MAX_LEN = 200

def test_preprocessing():
    """Test text preprocessing"""
    print("Testing text preprocessing...")
    
    # Sample word index (simplified)
    word_index = {
        'the': 1, 'movie': 2, 'was': 3, 'great': 4, 'terrible': 5,
        'amazing': 6, 'awful': 7, 'loved': 8, 'it': 9, 'boring': 10
    }
    
    # Test text
    test_text = "the movie was amazing"
    words = test_text.split()
    
    # Convert to indices
    sequence_data = [word_index.get(word, 0) for word in words]
    print(f"  Original text: {test_text}")
    print(f"  Word indices: {sequence_data}")
    
    # Pad sequence
    padded = sequence.pad_sequences([sequence_data], maxlen=MAX_LEN)
    print(f"  Padded shape: {padded.shape}")
    print("  ✓ Preprocessing works!\n")
    
    return True

def test_model_architecture():
    """Test model architecture"""
    print("Testing model architecture...")
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
    
    # Test SimpleRNN
    model_rnn = Sequential([
        Embedding(10000, 128, input_length=MAX_LEN),
        SimpleRNN(64),
        Dense(1, activation='sigmoid')
    ])
    print("  ✓ SimpleRNN architecture valid")
    
    # Test LSTM
    model_lstm = Sequential([
        Embedding(10000, 128, input_length=MAX_LEN),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    print("  ✓ LSTM architecture valid\n")
    
    return True

def test_prediction_pipeline():
    """Test prediction with random data"""
    print("Testing prediction pipeline...")
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense
    
    # Create a simple model
    model = Sequential([
        Embedding(10000, 128, input_length=MAX_LEN),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Test with random input
    test_input = np.random.randint(0, 10000, (1, MAX_LEN))
    prediction = model.predict(test_input, verbose=0)
    
    print(f"  Test prediction: {prediction[0][0]:.4f}")
    print("  ✓ Prediction pipeline works!\n")
    
    return True

def main():
    print("="*50)
    print("RNN vs LSTM Sentiment Analyzer - System Test")
    print("="*50)
    print()
    
    try:
        # Run tests
        test_preprocessing()
        test_model_architecture()
        test_prediction_pipeline()
        
        print("="*50)
        print("✓ ALL TESTS PASSED!")
        print("="*50)
        print()
        print("Your system is ready. Next steps:")
        print("1. Run 'python train_models.py' to train the models")
        print("2. Run 'python app.py' to start the server")
        print("3. Open http://localhost:5000 in your browser")
        
    except Exception as e:
        print("="*50)
        print("✗ TEST FAILED!")
        print("="*50)
        print(f"Error: {str(e)}")
        print("\nPlease check your dependencies:")
        print("pip install -r requirements.txt")

if __name__ == '__main__':
    main()
