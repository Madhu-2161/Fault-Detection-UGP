# Fault Detection & Classification for the Tennessee Eastman Process

This project focuses on building and comparing machine learning and deep learning models to detect and classify faults in the **Tennessee Eastman Process (TEP)**, a benchmark simulation of a chemical plant.

The primary goal is to build a robust system that can identify process abnormalities in real-time, reducing safety risks and financial losses.

## Project Structure

This project explores a two-stage approach:
1.  **Stage 1: Fault Detection (Binary)**
    * A binary classification model that determines *if* a fault is present (Class 0: Normal vs. Class 1: Fault).
2.  **Stage 2: Fault Classification (Multiclass)**
    * A multiclass classification model that identifies *which* specific fault has occurred (21 unique classes: 1 Normal + 20 Fault Types).

## 🗂️ Models in this Repository

This repository contains the notebooks for the full analysis:

* **`Fault_Detection_EDA.ipynb`**: Initial Exploratory Data Analysis and visualization.
* **`Decision_Tree.ipynb`**: Implementation of a Decision Tree classifier.
* **`KNN.ipynb`**: Implementation of a K-Nearest Neighbors classifier.
* **`RandomForest.ipynb`**: Implementation of a Random Forest classifier.
* **`XGBoost.ipynb`**: Implementation of an XGBoost classifier.
* **`Fault_Detection_Classification.ipynb`**: The main notebook containing the implementation of the advanced deep learning models (CNN and LSTM) discussed in the report.

---

## 🚀 Project Evolution & Key Findings

The final, optimized models were the result of a rigorous tuning process that solved several common data science challenges.

### Finding 1: Handling Extreme Class Imbalance

* **Problem:** The "Normal" (Class 0) was 18x rarer than "Fault" classes. Initial models achieved a "fake" 94% accuracy by simply guessing "Fault" every time, resulting in 0% recall for the "Normal" state.
* **Solution:** We implemented **`class_weight='balanced'`** in TensorFlow/Keras. This penalized the model 18x more for making a mistake on the rare class, forcing it to learn to identify the "Normal" state.

### Finding 2: Conquering Severe Overfitting

* **Problem:** Once the models started learning, they overfit almost immediately. The training accuracy would climb while the validation (test) accuracy would get *worse*.
* **Solution:** We implemented **`EarlyStopping`**. This callback monitored the `val_loss` and automatically stopped training, restoring the model weights from its single best-performing epoch.

### Finding 3: Architecture is Everything (1D vs. 2D)

We compared four different deep learning architectures. The results clearly show that for time-series data, **temporal context is the most important feature.**

| Model | Stage 2 Accuracy | Macro F1-Score | "No Fault" Recall | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **2D-CNN** | 49% | 0.50 | 27% | **Failed:** Loses critical temporal data. |
| **2D-CNN-LSTM** | 53% | 0.53 | 58% | **Failed:** Too complex, overfit instantly. |
| **LSTM (Pure)** | 60% | 0.62 | **81%** | **Strong Baseline:** Excellent at learning temporal patterns. |
| **1D-CNN-LSTM** | **🏆 63%** | **🏆 0.65** | 70% | **WINNER:** Best overall performance and balance. |

---

## 🏆 The Winning Model: 1D-CNN-LSTM

The best results were achieved with a **Hybrid 1D-CNN-LSTM** model. This architecture gets the best of both worlds:

1.  **1D-CNN (Feature Extractor):** The `Conv1D` layer scans the 10-step sequence, acting as a powerful filter to find small, predictive patterns *within* the sequence.
2.  **LSTM (Sequence Learner):** The `LSTM` layer receives these *abstract features* (not the raw data) and learns the complex *temporal relationships between them*.

This two-step process created the most accurate and balanced model for classifying all 21 process states.

## 🧑‍💻 Contributors

* Madhu Kumari
* Aditi Vishwas Kamble
* Piyush Singh
