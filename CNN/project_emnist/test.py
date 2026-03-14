
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(r'C:\Users\USER\Documents\PYTHON\MYDL\cnnn\New folder\emnist_lenet_model.h5')

# Test input - siyah resim
test_img = np.zeros((28, 28), dtype=np.float32)
test_img[10:18, 10:18] = 1.0  # Beyaz kare

print("VARIANT 1: transpose(0,2,1,3) + flip(axis=1)")
arr = test_img.reshape(1, 28, 28, 1)
arr = np.transpose(arr, (0, 2, 1, 3))
arr = np.flip(arr, axis=1)
pred = model.predict(arr, verbose=0)[0]
print(f"Top index: {np.argmax(pred)}\n")

print("VARIANT 2: transpose(0,2,1,3) + flip(axis=2)")
arr = test_img.reshape(1, 28, 28, 1)
arr = np.transpose(arr, (0, 2, 1, 3))
arr = np.flip(arr, axis=2)
pred = model.predict(arr, verbose=0)[0]
print(f"Top index: {np.argmax(pred)}\n")

print("VARIANT 3: No transpose, no flip")
arr = test_img.reshape(1, 28, 28, 1)
pred = model.predict(arr, verbose=0)[0]
print(f"Top index: {np.argmax(pred)}\n")

print("VARIANT 4: Only transpose(0,2,1,3)")
arr = test_img.reshape(1, 28, 28, 1)
arr = np.transpose(arr, (0, 2, 1, 3))
pred = model.predict(arr, verbose=0)[0]
print(f"Top index: {np.argmax(pred)}\n")

print("VARIANT 5: Flip axis=1, then transpose")
arr = test_img.reshape(1, 28, 28, 1)
arr = np.flip(arr, axis=1)
arr = np.transpose(arr, (0, 2, 1, 3))
pred = model.predict(arr, verbose=0)[0]
print(f"Top index: {np.argmax(pred)}\n")

print("VARIANT 6: Flip axis=2, then transpose")
arr = test_img.reshape(1, 28, 28, 1)
arr = np.flip(arr, axis=2)
arr = np.transpose(arr, (0, 2, 1, 3))
pred = model.predict(arr, verbose=0)[0]
print(f"Top index: {np.argmax(pred)}\n")

print("VARIANT 7: Only flip(axis=1)")
arr = test_img.reshape(1, 28, 28, 1)
arr = np.flip(arr, axis=1)
pred = model.predict(arr, verbose=0)[0]
print(f"Top index: {np.argmax(pred)}\n")

print("VARIANT 8: Only flip(axis=2)")
arr = test_img.reshape(1, 28, 28, 1)
arr = np.flip(arr, axis=2)
pred = model.predict(arr, verbose=0)[0]
print(f"Top index: {np.argmax(pred)}\n")


