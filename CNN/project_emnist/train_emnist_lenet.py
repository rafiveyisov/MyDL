# train_emnist_lenet_local.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class LocalDataLoader:
    def __init__(self, train_path, test_path):
        """
        Lokal dataset-ləri yükləyir
        train_path: train CSV faylının yolu
        test_path: test CSV faylının yolu
        """
        self.train_path = train_path
        self.test_path = test_path
        print(f"📂 Train path: {train_path}")
        print(f"📂 Test path: {test_path}")
        
        # EMNIST datasetində sinif sayı (0-9, A-Z, a-z) = 62
        self.num_classes = 62
        
    def load_csv_data(self, file_path, nrows=None):
        """CSV faylından data yükləyir"""
        print(f"📊 Loading: {file_path}")
        if nrows:
            df = pd.read_csv(file_path, nrows=nrows)
        else:
            df = pd.read_csv(file_path)
        
        # İlk sütun label, qalanları pixel dəyərləri
        labels = df.iloc[:, 0].values
        pixels = df.iloc[:, 1:].values
        
        return pixels, labels
    
    def preprocess_data(self, pixels, labels):
        """Data-nı model üçün hazırlayır"""
        # Pixel dəyərlərini normalize et (0-255 -> 0-1)
        pixels = pixels.astype('float32') / 255.0
        
        # Şəkilləri 28x28 formatına sal
        pixels = pixels.reshape(-1, 28, 28, 1)
        
        # EMNIST üçün xüsusi transformasiya (transpose + flip)
        # Bu, şəkilləri düzgün oriyentasiyaya gətirir
        pixels = np.transpose(pixels, (0, 2, 1, 3))
        pixels = np.flip(pixels, axis=1)
        
        # One-hot encoding
        labels_one_hot = tf.keras.utils.to_categorical(labels, self.num_classes)
        
        return pixels, labels_one_hot
    
    def load_dataset(self, train_nrows=100000, test_nrows=20000):
        """
        Dataset-i yükləyir və preprocessing edir
        train_nrows: neçə train sample istifadə olunsun (hamısı üçün None)
        test_nrows: neçə test sample istifadə olunsun (hamısı üçün None)
        """
        # Train data-nı yüklə
        print("\n🔄 Loading training data...")
        train_pixels, train_labels = self.load_csv_data(self.train_path, train_nrows)
        
        # Test data-nı yüklə
        print("🔄 Loading test data...")
        test_pixels, test_labels = self.load_csv_data(self.test_path, test_nrows)
        
        # Preprocessing
        print("🔄 Preprocessing training data...")
        trainX, trainY = self.preprocess_data(train_pixels, train_labels)
        
        print("🔄 Preprocessing test data...")
        testX, testY = self.preprocess_data(test_pixels, test_labels)
        
        print(f"\n✅ Train shape: {trainX.shape}")
        print(f"✅ Train labels shape: {trainY.shape}")
        print(f"✅ Test shape: {testX.shape}")
        print(f"✅ Test labels shape: {testY.shape}")
        print(f"✅ Number of classes: {self.num_classes}")
        
        return trainX, trainY, testX, testY

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if keras.backend.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # First CONV => RELU => POOL
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # Second CONV => RELU => POOL
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # FC => RELU
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        
        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

def create_label_mapping():
    """EMNIST label mapping yaradır (0-9, A-Z, a-z)"""
    mapping = {}
    # Rəqəmlər (0-9)
    for i in range(10):
        mapping[i] = str(i)
    # Böyük hərflər (A-Z)
    for i in range(26):
        mapping[10 + i] = chr(ord('A') + i)
    # Kiçik hərflər (a-z)
    for i in range(26):
        mapping[36 + i] = chr(ord('a') + i)
    
    # Mapping-i saxla
    mapping_df = pd.DataFrame(list(mapping.items()), columns=['label', 'character'])
    mapping_df.to_csv('label_mapping.csv', index=False)
    print("✅ Label mapping saved to 'label_mapping.csv'")
    
    return mapping

def plot_sample_images(trainX, trainY, label_map, num_samples=10):
    """Nümunə şəkilləri göstər"""
    indices = np.random.choice(len(trainX), num_samples, replace=False)
    
    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i+1)
        plt.imshow(trainX[idx].reshape(28, 28), cmap='gray')
        label = np.argmax(trainY[idx])
        plt.title(f"Label: {label_map[label]}", fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.show()
    print("✅ Sample images saved to 'sample_images.png'")

def plot_training_history(H):
    """Təlim tarixçəsini qrafiklə göstər"""
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(H.history['accuracy'], label='train_accuracy', linewidth=2)
    plt.plot(H.history['val_accuracy'], label='val_accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(H.history['loss'], label='train_loss', linewidth=2)
    plt.plot(H.history['val_loss'], label='val_loss', linewidth=2)
    plt.title('Model Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    print("📊 Training history saved to 'training_history.png'")

def main():
    print("="*60)
    print("🔬 EMNIST LENET TRAINING with LOCAL DATASET")
    print("="*60)
    
    # Dataset yolları
    train_path = r"C:\Users\USER\Documents\Datasets\emnist-byclass-train.csv"
    test_path = r"C:\Users\USER\Documents\Datasets\emnist-byclass-test.csv"
    
    # Faylların mövcudluğunu yoxla
    if not os.path.exists(train_path):
        print(f"❌ Train file not found: {train_path}")
        return
    if not os.path.exists(test_path):
        print(f"❌ Test file not found: {test_path}")
        return
    
    # 1. Data yüklə
    print("\n📂 Step 1: Loading local datasets...")
    data_loader = LocalDataLoader(train_path, test_path)
    
    # İstəsəniz, nümunə sayını məhdudlaşdıra bilərsiniz
    # train_nrows=50000 (50k sample) və ya None (hamısı)
    trainX, trainY, testX, testY = data_loader.load_dataset(
        train_nrows=100000,  # 100k sample
        test_nrows=20000     # 20k sample
    )
    
    # 2. Label mapping yarat
    print("\n🔤 Step 2: Creating label mapping...")
    label_map = create_label_mapping()
    
    # 3. Nümunə şəkillərə bax
    print("\n👀 Step 3: Displaying sample images...")
    plot_sample_images(trainX, trainY, label_map)
    
    # 4. Modeli qur
    print("\n🧠 Step 4: Building LeNet model...")
    num_classes = data_loader.num_classes
    model = LeNet.build(width=28, height=28, depth=1, classes=num_classes)
    
    # 5. Modeli compile et
    print("\n⚙️ Step 5: Compiling model...")
    opt = SGD(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.summary()
    
    # 6. Modeli öyrət
    print("\n🚀 Step 6: Training model (this may take 10-15 minutes)...")
    H = model.fit(
        trainX, trainY,
        validation_data=(testX, testY),
        batch_size=128,
        epochs=20,
        verbose=1
    )
    
    # 7. Təlim qrafiklərini göstər
    print("\n📈 Step 7: Plotting training history...")
    plot_training_history(H)
    
    # 8. Modeli qiymətləndir
    print("\n📊 Step 8: Evaluating model...")
    predictions = model.predict(testX, batch_size=128)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(testY, axis=1)
    
    # Dəqiqliyi hesabla
    accuracy = np.mean(y_pred == y_true)
    print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")
    
    # 9. Modeli saxla
    print("\n💾 Step 9: Saving model...")
    model.save("emnist_lenet_model.h5")
    print("✅ Model saved to 'emnist_lenet_model.h5'")
    
    # 10. Test nəticələrini göstər
    print("\n🔍 Step 10: Testing with random samples...")
    num_samples = 10
    indices = np.random.choice(len(testX), num_samples, replace=False)
    
    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(indices):
        img = testX[idx]
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        
        plt.subplot(2, 5, i+1)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {label_map[true_label]}\nPred: {label_map[pred_label]}", 
                 color=color, fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_samples.png')
    plt.show()
    print("✅ Test samples saved to 'test_samples.png'")
    
    # 11. Classification report
    print("\n📋 Classification Report (first 10 classes):")
    # Sadəcə ilk 10 sinif üçün report (qarışıq olmasın deyə)
    mask = y_true < 10
    if np.any(mask):
        print(classification_report(y_true[mask], y_pred[mask], 
                                  labels=range(10), zero_division=0))
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📁 Generated files:")
    print("   - emnist_lenet_model.h5  (trained model)")
    print("   - label_mapping.csv       (character mapping)")
    print("   - sample_images.png        (training samples)")
    print("   - training_history.png     (accuracy/loss plots)")
    print("   - test_samples.png         (test predictions)")

if __name__ == "__main__":
    main()