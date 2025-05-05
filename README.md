# 🧠🖼️ Image Captioning AI (ResNet + LSTM)

This project demonstrates a simple image captioning system that combines **computer vision** and **natural language processing** using a pre-trained **ResNet50** model for feature extraction and an **LSTM-based decoder** to generate captions.

> ⚠️ This is a demo with random weights and dummy captions. For real performance, train the model on datasets like **Flickr8k** or **MSCOCO**.

---

## 🔧 Features

- Extracts visual features from any uploaded image using **ResNet50**
- Dynamically selects images using a file dialog
- Generates captions using a dummy tokenizer and LSTM decoder
- Clean, single-script implementation (no dependencies on external datasets)

---

## 📦 Requirements

Install dependencies:

```bash
pip install tensorflow pillow
```

---

## 🚀 How to Run

1. Save the script as `image_captioning.py`
2. Run the script:

```bash
python image_captioning.py
```

3. A file dialog will appear. Select a `.jpg` or `.png` image.
4. The console will display a dummy generated caption like:

```
Selected file: C:/Users/.../dog.jpg
Generated Caption: a dog playing frisbee
```

---

## 📁 File Structure

```
image_captioning.py   # Main script
README.md             # Project guide
```

---

## 🧠 How It Works

1. **CNN Encoder**: Uses ResNet50 (pre-trained on ImageNet) to extract a 2048-dim feature vector.
2. **LSTM Decoder**: Generates captions based on extracted image features and embedded word sequences.
3. **Tokenizer**: Dummy tokenizer mimicking a trained vocabulary.
4. **End-to-End Flow**:
   - Load image → Extract features → Predict next word → Build full caption

---

## ✅ To-Do for Full Version

- [ ] Replace dummy tokenizer with trained tokenizer
- [ ] Load and preprocess dataset like Flickr8k
- [ ] Train the decoder model on image-caption pairs
- [ ] Use evaluation metrics like BLEU, CIDEr

---

## 📷 Sample Output

```
Selected file: dog.jpg  
Generated Caption: a dog playing frisbee
```

---

## 📝 License

Free to use for educational purposes.

---
