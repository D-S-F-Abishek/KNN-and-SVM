Banknote Authentication using KNN & SVM
Project Overview

This project focuses on classifying banknotes as authentic or fake using machine learning algorithms K-Nearest Neighbors (KNN) and Support Vector Machine (SVM). The aim is to provide a practical implementation to explore the effects of feature scaling, hyperparameters, and model evaluation on real-world numerical data.

Dataset

Source: UCI Banknote Authentication Dataset

Description: The dataset contains 1,372 samples of banknotes with features extracted from wavelet-transformed images.

Features:

Variance – spread of pixel intensity in wavelet-transformed image

Skewness – asymmetry of pixel intensity distribution

Kurtosis – “peakedness” of the pixel intensity distribution

Entropy – randomness/complexity of texture

Target: class → 0: Fake, 1: Authentic

Format: Text file with 4 numeric features and 1 label per row.

Project Steps

Data Loading & Exploration

Loaded dataset from text file.

Checked dataset shape, head, and class distribution.

Data Preprocessing

Split dataset into features (X) and target (y).

Split data into training (80%) and testing (20%).

Applied standard scaling to numeric features for KNN & SVM.

Model Training & Prediction

KNN Classifier

n_neighbors=5

Trained on scaled training data, predicted test labels

SVM Classifier

kernel='linear'

Trained on scaled training data, predicted test labels

Evaluation Metrics

Accuracy – overall correctness of predictions

Confusion Matrix – detailed insight into misclassifications

Classification Report – precision, recall, F1-score

ROC-AUC Score – area under ROC curve to evaluate classifier performance

Visualization

Confusion matrices visualized using heatmaps

ROC curves to compare KNN and SVM performance

Results
Model	Accuracy	ROC-AUC
KNN (k=5)	99%	0.99
SVM (Linear)	99%	1.00

Both models performed exceptionally well due to the clean, well-structured dataset.

Slight differences observed in decision scores and misclassified samples.

Learnings

Feature Scaling: Critical for distance-based algorithms (KNN) and margin-based algorithms (SVM).

KNN: Performance depends on k value and distance metric. Too small k may overfit; too large k may underfit.

SVM: Linear kernel sufficient for this dataset; decision scores help compute ROC-AUC.

Model Evaluation: Confusion matrices and ROC curves provide insights beyond simple accuracy.

Real-World Insight: Even a small set of engineered features (variance, skewness, kurtosis, entropy) can effectively distinguish real vs fake banknotes.

Next Steps / Extensions

Experiment with different K values and distance metrics for KNN.

Explore non-linear SVM kernels (RBF, polynomial) for more complex datasets.

Implement cross-validation for robust evaluation.

Deploy as a real-time prediction tool for banknote authentication with live inputs.

How to Run

Clone the repository:

git clone <repository-url>


Install required libraries:

pip install pandas numpy scikit-learn matplotlib seaborn


Place the dataset file data_banknote_authentication.txt in the same folder.

Run the notebook or Python script:

python banknote_authentication.py


Visualizations and metrics will be displayed for KNN and SVM classifiers.

References

UCI Machine Learning Repository – Banknote Authentication Dataset

Scikit-learn documentation – KNN & SVM
