# Üsul 1: Modeli lokalda saxlayıb sonra istifadə etmək
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch

# Model adı (Hugging Face-də olan EMNIST modeli)
model_name = "ShadowProgrammer/EMNISTClassifier"  # Nümunə model adı

# Modeli və feature extractor-u yüklə
model             = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Modeli lokalda saxla (əgər .h5 formatında saxlamaq istəsəniz)
model.save_pretrained("./emnist_hf_model")
feature_extractor.save_pretrained("./emnist_hf_model")