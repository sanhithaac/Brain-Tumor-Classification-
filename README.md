# 🧠 Brain Tumor Classification using Deep Learning  
### 🩺 MRI-Based Medical Image Analysis

---

## 🚀 Overview
This project focuses on **brain tumor classification from MRI scans** using **deep learning and transfer learning** techniques.  
Multiple convolutional neural network (CNN) architectures were implemented and evaluated to identify the most effective model for accurate tumor detection.

A total of **five architectures** were tested, with **GoogLeNet achieving the best overall performance** among them.

---

## 🧠 Models Evaluated
The following pre-trained CNN architectures were fine-tuned and compared:

- 🟢 **GoogLeNet (Best Performing Model)**
- 🔵 VGGNet
- 🟣 ResNet
- 🟠 AlexNet
- ⚪ Baseline CNN (for comparison)

All models were trained using transfer learning on MRI image patches for **binary classification (Tumor vs No Tumor)**.

---

## ⭐ Key Features
- MRI-based brain tumor classification
- Transfer learning with multiple CNN architectures
- Patch-based image preprocessing
- Comparative performance evaluation
- GoogLeNet identified as best-performing model
- GPU-accelerated training (CUDA support)
- Clean and reproducible training pipeline

---

## 📊 Dataset
- **Modality:** Brain MRI scans  
- **Task:** Binary classification (Tumor / No Tumor)  
- **Input Type:** Patch-based image samples  
- **Preprocessing Steps:**
  - Grayscale normalization
  - Patch extraction
  - Resizing to CNN input dimensions
  - Class balancing strategies

---

## 🔬 Methodology

### 🧩 Preprocessing
- MRI scans divided into fixed-size patches
- Tumor presence determined using annotation overlap
- Grayscale images converted to 3-channel format for CNN input
- Normalization applied for stable training

---

### 🤖 Deep Learning Pipeline
1. Load pre-trained CNN architecture
2. Replace final classification layer
3. Fine-tune on MRI patches
4. Evaluate on validation data
5. Compare performance across architectures

---

## 📈 Results Summary
- ✅ Deep learning significantly outperformed traditional approaches
- ✅ Transfer learning reduced training time
- ✅ GoogLeNet achieved the **highest validation accuracy**
- ✅ Models generalized well despite class imbalance

---

## 🧠 Why GoogLeNet Performed Best
- Inception modules capture multi-scale features
- Efficient parameter usage
- Strong generalization on medical images
- Balanced depth and computational cost

---

## 🛠️ Tech Stack
- **Language:** Python
- **Framework:** PyTorch
- **Models:** GoogLeNet, VGGNet, ResNet, AlexNet
- **Libraries:** NumPy, OpenCV, scikit-learn, matplotlib
- **Environment:** Jupyter Notebook
- **Hardware:** CUDA-enabled GPU (recommended)

---

## 🔮 Future Improvements
- Add Grad-CAM visualizations for explainability
- Address class imbalance using weighted loss
- Extend to multi-class tumor classification
- Deploy as a web-based diagnostic tool

---

## ⚠️ Disclaimer
This project is intended for **educational and research purposes only** and should **not** be used for real-world medical diagnosis.

---

⭐ *If you find this project useful, consider starring the repository!*
