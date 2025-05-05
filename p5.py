import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Add
import numpy as np
import os
from tkinter import filedialog
from tkinter import Tk
import sys

# ---------------------------
# Step 1: Feature Extractor (ResNet-50)
# ---------------------------
def build_feature_extractor():
    base_model = ResNet50(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model

def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except FileNotFoundError:
        print(f"[ERROR] File not found: {img_path}")
        sys.exit(1)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x, verbose=0)
    return feature

# ---------------------------
# Step 2: Tokenizer and dummy captions (simulate training data)
# ---------------------------
tokenizer = tf.keras.preprocessing.text.Tokenizer()
dummy_captions = [
    "startseq a dog playing frisbee endseq",
    "startseq a man riding a horse endseq",
    "startseq children playing in park endseq",
    "startseq a cat sitting on the table endseq",
    "startseq an airplane flying in the sky endseq",
    "startseq a person riding a bicycle endseq",
    "startseq a group of people walking endseq"
]
tokenizer.fit_on_texts(dummy_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in dummy_captions)

# ---------------------------
# Step 3: Decoder model (CNN + LSTM)
# ---------------------------
def build_captioning_model():
    inputs1 = Input(shape=(2048,))
    fe1 = Dense(256, activation='relu')(inputs1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256)(se1)

    decoder1 = Add()([fe1, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# ---------------------------
# Step 4: Caption generation logic
# ---------------------------
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.replace("startseq", "").replace("endseq", "").strip()
    # If empty, return placeholder
    return final if final else "a dog playing frisbee"

# ---------------------------
# Step 5: Main logic — select image and generate caption
# ---------------------------
if __name__ == "__main__":
    # Load ResNet model
    fe_model = build_feature_extractor()

    # Prompt user to select image
    print("Please select an image file...")
    Tk().withdraw()
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    
    if not img_path:
        print("[ERROR] No file selected. Exiting.")
        sys.exit(1)

    print(f"Selected file: {img_path}")

    # Extract image features
    photo = extract_features(img_path, fe_model)

    # Build the captioning model (note: no training, random weights)
    model = build_captioning_model()

    # Generate and print caption (fallback to default if blank)
    print("Generating caption (dummy model, demo only)...")
    caption = generate_caption(model, tokenizer, photo, max_length)
    print("Generated Caption:", caption)
