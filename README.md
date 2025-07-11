# Handwritten Roman Numeral Recognition 🏛️🔢

A deep learning project that recognizes handwritten Roman numerals (I to X) using MobileNetV2 and deploys the model through a Flask-based web application.

---

## 📌 Introduction

Recognizing handwritten Roman numerals is a challenging computer vision task due to high similarity between characters and diverse handwriting styles. This project leverages transfer learning with **MobileNetV2** to build a robust image classifier capable of identifying Roman numerals from I to X. The trained model is deployed via a web interface using **Flask**.

---

## 🎯 Objectives

- Classify handwritten Roman numerals (I to X) using a CNN-based approach.
- Utilize **transfer learning** with MobileNetV2 to handle small datasets effectively.
- Evaluate performance using accuracy metrics and a confusion matrix.
- Deploy the trained model as an interactive web application using Flask.

---

## 📂 Dataset

- Contains images of handwritten Roman numerals labeled from I to X.
- Each numeral class is stored in a separate folder (e.g., `i/`, `ii/`, ..., `x/`).
- Dataset is split into:
  - Training Set
  - Validation Set
  - Test Set

---

## 🧠 Model Architecture & Training

### 🔧 Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- Custom Layers:
  ```python
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(64, activation='relu')(x)
  predictions = Dense(len(class_labels), activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
