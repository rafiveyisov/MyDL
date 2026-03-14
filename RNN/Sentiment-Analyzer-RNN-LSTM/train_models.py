import numpy as np
from tensorflow.keras.datasets      import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models        import Sequential
from tensorflow.keras.layers        import Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.callbacks      import EarlyStopping
import pickle

MAX_FEATURES = 10000 
MAX_LEN = 200  
BATCH_SIZE = 32
EPOCHS = 10

print("Loading IMDB dataset")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

word_index = imdb.get_word_index()
with open('word_index.pkl', 'wb') as f:
    pickle.dump(word_index, f)
print("Word index saved!")

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)


# RNN

model_rnn = Sequential([
                            Embedding(MAX_FEATURES, 128, input_length=MAX_LEN),
                            SimpleRNN(64, dropout=0.2),
                            Dense(1, activation='sigmoid')
                        ])

model_rnn.compile(
                        optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )

model_rnn.summary()

history_rnn = model_rnn.fit(
                                x_train, y_train,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                validation_split=0.2,
                                callbacks=[early_stop],
                                verbose=1
                            )

rnn_loss, rnn_accuracy = model_rnn.evaluate(x_test, y_test, verbose=0)
print(f"\nSimpleRNN Test Accuracy: {rnn_accuracy:.4f}")

model_rnn.save('simple_rnn_model.h5')
print("SimpleRNN model saved as 'simple_rnn_model.h5'")


# LSTM 


print('  ')
model_lstm = Sequential([
                            Embedding(MAX_FEATURES, 128, input_length=MAX_LEN),
                            LSTM(64, dropout=0.2),
                            Dense(1, activation='sigmoid')
                        ])

model_lstm.compile(
                        optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )

model_lstm.summary()

history_lstm = model_lstm.fit(
                                    x_train, y_train,
                                    batch_size=BATCH_SIZE,
                                    epochs=EPOCHS,
                                    validation_split=0.2,
                                    callbacks=[early_stop],
                                    verbose=1
                                )

lstm_loss, lstm_accuracy = model_lstm.evaluate(x_test, y_test, verbose=0)
print(f"\nLSTM Test Accuracy: {lstm_accuracy:.4f}")

model_lstm.save('lstm_model.h5')
print("LSTM model saved as 'lstm_model.h5'")



print(f"SimpleRNN Test Accuracy: {rnn_accuracy:.4f}")
print(f"LSTM Test Accuracy:      {lstm_accuracy:.4f}")
print(f"Improvement:             {(lstm_accuracy - rnn_accuracy):.4f}")
