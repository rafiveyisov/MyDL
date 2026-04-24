# tensorboard_demo.py
import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# 1. Məlumat yaratmaq (süni dataset)
def create_dataset():
    np.random.seed(42)
    n_samples = 1000
    
    X = np.random.randn(n_samples, 10)  # 10 xüsusiyyət
    y = X[:, 0] * 2.5 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1
    
    # Train/test split
    split_idx = int(n_samples * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return (X_train, y_train), (X_test, y_test)

# 2. Log qovluğu üçün funksiya
def get_run_logdir(root_logdir="my_logs"):
    """Hər işləyiş üçün unikal qovluq yaradır"""
    run_id = datetime.now().strftime("run_%Y_%m_%d_%H_%M_%S")
    return Path(root_logdir) / run_id

# 3. Model qurmaq funksiyası
def build_model(learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Regression üçün
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']  # Mean Absolute Error
    )
    
    return model

# 4. Əsas təlim funksiyası
def train_model(learning_rate=0.001, epochs=50, run_name=None):
    # Məlumatları yüklə
    (X_train, y_train), (X_val, y_val) = create_dataset()
    
    # Modeli qur
    model = build_model(learning_rate)
    
    # Log qovluğunu yarat
    if run_name:
        run_logdir = Path("my_logs") / run_name
    else:
        run_logdir = get_run_logdir()
    
    # Qovluğu yarat
    run_logdir.mkdir(parents=True, exist_ok=True)
    print(f"Log qovluğu: {run_logdir}")
    
    # TensorBoard callback yarat
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=run_logdir,
        histogram_freq=1,          # Hər epoch-da histogramları yaz
        write_graph=True,          # Qrafiki yaz
        write_images=True,         # Şəkilləri yaz
        update_freq='epoch',       # Hər epoch-da yaz
        profile_batch=(100, 200)   # Profil analizi
    )
    
    # Early stopping callback
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=10,
        restore_best_weights=True
    )
    
    # Modeli təlim et
    print(f"\nTəlim başladı (Learning Rate: {learning_rate})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1,
        callbacks=[tensorboard_cb, early_stopping_cb]
    )
    
    # Qiymətləndirmə
    test_loss, test_mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nTest Nəticələri:")
    print(f"Loss (MSE): {test_loss:.4f}")
    print(f"MAE: {test_mae:.4f}")
    
    return model, history, str(run_logdir)

# 5. TensorBoard-u başlatmaq üçün funksiya
def start_tensorboard(log_dir="my_logs"):
    """TensorBoard serverini başladır"""
    import subprocess
    import webbrowser
    import time
    
    # Log qovluğunu yoxla
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Xəta: '{log_dir}' qovluğu tapılmadı!")
        print("Əvvəlcə modeli təlim etməlisiniz.")
        return
    
    # TensorBoard-u başlat
    print(f"\nTensorBoard serveri başladılır...")
    print(f"Log qovluğu: {log_path.absolute()}")
    print("\nBrauzerdə açmaq üçün: http://localhost:6006")
    print("Dayandırmaq üçün: Ctrl+C")
    
    try:
        # TensorBoard prosesini başlat
        process = subprocess.Popen(
            ["tensorboard", "--logdir", str(log_path.absolute()), "--port", "6006"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Brauzeri aç
        time.sleep(3)  # Serverin başlamağı üçün gözlə
        webbrowser.open("http://localhost:6006")
        
        print("\nServer işləyir. Dayandırmaq üçün Enter düyməsini basın...")
        input()  # İstifadəçinin Enter basmasını gözlə
        
        # Prosesi dayandır
        process.terminate()
        print("TensorBoard serveri dayandırıldı.")
        
    except KeyboardInterrupt:
        print("\nTensorBoard serveri dayandırıldı.")
    except Exception as e:
        print(f"Xəta: {e}")

# 6. Əsas proqram
def main():
    print("=" * 60)
    print("TENSORBOARD DEMO - VS Code")
    print("=" * 60)
    
    # my_logs qovluğunu təmizlə (seçimlik)
    if Path("my_logs").exists():
        print("\nQeyd: 'my_logs' qovluğu mövcuddur.")
        print("Əvvəlki təcrübələr saxlanılacaq.")
    
    while True:
        print("\n" + "=" * 60)
        print("MENYU:")
        print("1. Modeli təlim et (learning_rate=0.001)")
        print("2. Modeli təlim et (learning_rate=0.002)")
        print("3. Fərqli parametrlə təlim et")
        print("4. TensorBoard-u başlat")
        print("5. Log qovluqlarını təmizlə")
        print("6. Çıxış")
        
        choice = input("\nSeçiminiz (1-6): ").strip()
        
        if choice == "1":
            print("\n" + "-" * 40)
            model, history, log_dir = train_model(learning_rate=0.001, epochs=30)
            print(f"\nTəlim tamamlandı! Loglar: {log_dir}")
            
        elif choice == "2":
            print("\n" + "-" * 40)
            model, history, log_dir = train_model(learning_rate=0.002, epochs=30)
            print(f"\nTəlim tamamlandı! Loglar: {log_dir}")
            
        elif choice == "3":
            print("\n" + "-" * 40)
            try:
                lr = float(input("Learning rate daxil edin (məs: 0.005): "))
                epochs = int(input("Epoch sayı daxil edin (məs: 50): "))
                run_name = input("Run adı (boş buraxın avtomatik yaratsın): ").strip()
                
                if run_name == "":
                    run_name = None
                    
                model, history, log_dir = train_model(
                    learning_rate=lr, 
                    epochs=epochs,
                    run_name=run_name
                )
                print(f"\nTəlim tamamlandı! Loglar: {log_dir}")
            except ValueError:
                print("Xəta: Düzgün dəyər daxil edin!")
                
        elif choice == "4":
            print("\n" + "-" * 40)
            start_tensorboard()
            
        elif choice == "5":
            print("\n" + "-" * 40)
            confirm = input("Bütün logları silmək istədiyinizə əminsiniz? (y/N): ")
            if confirm.lower() == 'y':
                import shutil
                if Path("my_logs").exists():
                    shutil.rmtree("my_logs")
                    print("'my_logs' qovluğu silindi.")
                else:
                    print("'my_logs' qovluğu tapılmadı.")
                    
        elif choice == "6":
            print("\nProqramdan çıxılır...")
            break
            
        else:
            print("Yanlış seçim! 1-6 arası rəqəm daxil edin.")

# 7. Əlavə yardımçı skript
if __name__ == "__main__":
    main()