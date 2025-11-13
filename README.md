# Acoustic Engine Fault Detection
Machine learning pipeline for classifying electric engine conditions using audio-based feature extraction (Amplitude Spectrum, VGGish, OpenL3, Wav2Vec 2.0).

---

## 📌 Overview

This project explores the use of acoustic signals and machine learning to detect the operational state of electric motors. Using the IDMT-ISA-Electric-Engine dataset, we evaluate four feature extraction techniques and ten classification models to determine which combinations most effectively identify engines as:

- **Good**
- **Heavy Load**
- **Broken**

The pipeline covers preprocessing, data augmentation, feature extraction, PCA transformation, model training, and final performance evaluation.

---

## 📂 Dataset

**Dataset:** IDMT-ISA-Electric-Engine  
**Total Samples:** 2,378 audio clips (3 seconds each)  
**Sampling Rate:** 44.1 kHz  
**Classes:** Good, Heavy Load, Broken  

The dataset is split into:

- **Training Set:** clean audio  
- **Test Set:** noisy audio with talking, white noise, atmospheric noise, and stress-test interference

To reduce the domain gap between clean training audio and noisy testing audio, we applied noise augmentation using:

- Gaussian noise  
- Conversational noise  
- Atmospheric noise  
- Crowd noise  
- Traffic noise  

Noise levels were randomized using SNR between **10–20 dB**.

---

## 🔊 Feature Extraction Methods

### **Amplitude Spectrum**
- Classical FFT-based representation  
- Captures frequency energy distribution  
- Lightweight and surprisingly effective  

### **VGGish**
- CNN-based audio embedding model  
- Pretrained on YouTube-8M  
- Input: log-Mel spectrogram  
- Output: 384-dimensional embedding per clip  

### **OpenL3**
- Self-supervised audio representation model  
- Trained on AudioSet  
- Input: log-Mel spectrogram  
- Output: 1536-dimensional embedding per clip  
- **Best performing feature set overall**

### **Wav2Vec 2.0**
- Transformer-based raw waveform model  
- Pretrained for speech  
- Output: 768-dimensional contextual embeddings (averaged per clip)

---

## 🧠 Models Trained

Each feature extraction method was evaluated using ten machine-learning models:

- Multinomial Logistic Regression  
- Linear Discriminant Analysis (LDA)  
- Quadratic Discriminant Analysis (QDA)  
- K-Nearest Neighbors (KNN)  
- Decision Tree (with pruning)  
- Bagging  
- Random Forest  
- Support Vector Machines (Linear and Radial)  
- XGBoost  

**Dimensionality Reduction:** PCA with 95% variance retained  
**Cross-Validation:** 5-fold for hyperparameter tuning  

---

## 📊 Results Summary

### **🏆 Best Feature Set: OpenL3**
- **Linear SVM achieved 98.37% accuracy**  
- Consistent high performance across all major linear classifiers  
- Most linearly separable embedding

### **Amplitude Spectrum**
- QDA achieved **94.90% accuracy**  
- Strong performance from tree-based models (RF, XGBoost)

### **Wav2Vec 2.0**
- QDA achieved **85.11% accuracy**  
- Good generalization despite being trained for speech tasks

### **VGGish**
- Most inconsistent feature set  
- LDA (82.14%) and Linear SVM (76.50%) performed best

---

## 📌 Key Insights

- **OpenL3** produces the most discriminative and stable audio embeddings.  
- **FFT-based Amplitude Spectrum** features remain competitive and computationally efficient.  
- **Wav2Vec 2.0** generalizes surprisingly well to non-speech mechanical sounds.  
- **VGGish** is sensitive to noise and benefits from fine-tuning.  
- Distance-based models (KNN, RBF SVM) degrade in high-dimensional spaces.  
- Ensemble models (Random Forest, XGBoost) consistently perform well across features.

---

## 🧪 Evaluation Metrics

All models were evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

The full report includes tables, plots, and confusion matrices for each model-feature combination.

---

## 🏁 Conclusion

This project demonstrates that audio-based engine fault detection is highly effective when advanced audio embeddings or well-engineered spectral features are paired with the right machine-learning models.

- **OpenL3 + Linear SVM** gave near-perfect performance.  
- Classical **Amplitude Spectrum** features performed strongly with QDA and ensemble models.  
- **Wav2Vec 2.0** provided strong, balanced performance.  
- **VGGish** was the least reliable without fine-tuning.

**Future Work:**
- Hybrid feature fusion  
- End-to-end CNN/CRNN training  
- Temporal modeling (LSTM, GRU, TCN)  
- Real-time deployment  
- Unsupervised anomaly detection for unlabeled industrial faults

---

## 👥 Authors

- **Abdullah Salmeh** – b00093434  
- **Mohamad Chehab** – b00090578  
- **Yousef Irshaid** – b00093447  

Course: **STA 401 – Introduction to Data Mining**  
Instructor: **Dr. Ayman Alzaatreh**  
Semester: **Spring 2025**

