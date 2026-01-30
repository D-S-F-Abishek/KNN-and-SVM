💵 Banknote Authentication using KNN & SVM

📌 Project Overview

This project implements machine learning models to classify banknotes as authentic or fake using the Banknote Authentication dataset.

Algorithms used: K-Nearest Neighbors (KNN) & Support Vector Machine (SVM)

Goal: Learn feature scaling, model training, evaluation, and hyperparameter tuning

Outcome: Compare model performance and understand how ML distinguishes fake vs real banknotes

🗂 Dataset Description
Feature	Type	Description
variance	Numeric	Spread of pixel intensity after wavelet transform
skewness	Numeric	Asymmetry of pixel intensity distribution
kurtosis	Numeric	“Peakedness” of the pixel intensity distribution
entropy	Numeric	Randomness / complexity of texture
class	Binary	0: Fake, 1: Authentic

Samples: 1,372

Source: [Banknote Authentication Dataset – UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)


Format: Text file, 4 features + 1 label per row

🧹 Data Preprocessing

Feature Separation: Split into X (features) and y (target)

Train/Test Split: 80% training, 20% testing

Feature Scaling: StandardScaler applied to normalize numeric features

⚠️ Note: Feature scaling is critical for KNN (distance-based) and SVM (margin-based) models.

🔍 Exploratory Analysis

Class Distribution:

Authentic (1): 762
Fake (0): 610


Dataset is clean with no missing values; features are numeric and ready for ML algorithms

🤖 Modeling Approach
1️⃣ KNN Classifier

n_neighbors = 5

Euclidean distance

Trained on scaled features

Predicted labels on test set

2️⃣ SVM Classifier

Linear kernel (kernel='linear')

Trained on scaled features

Predicted using decision scores for ROC-AUC

📊 Results
Model	Accuracy	ROC-AUC
KNN (k=5)	99%	0.99
SVM (Linear)	99%	1.00

Both models performed exceptionally well due to clear feature separation

Minor differences observed in decision scores and misclassifications

📈 Evaluation Metrics

Confusion Matrix – KNN

Confusion Matrix – SVM

Classification Report (SVM Example)

Metric	Fake (0)	Authentic (1)
Precision	0.99	1.00
Recall	0.99	0.99
F1-score	0.99	0.99
💡 Learnings

Feature Scaling: Essential for KNN and SVM performance

KNN: Performance depends on k and distance metric; overfitting/underfitting is visible with small/large k

SVM: Linear kernel sufficient for this dataset; decision scores help evaluate ROC-AUC

Model Evaluation: Confusion matrices and ROC curves provide insight beyond accuracy

Feature Engineering: Even 4 engineered features (variance, skewness, kurtosis, entropy) are sufficient for highly accurate classification

🚀 Next Steps / Extensions

Experiment with different K values and distance metrics in KNN

Test non-linear kernels (RBF, polynomial) in SVM

Implement cross-validation for robust evaluation

Deploy a GUI to input banknote features and predict authenticity in real-time

⚙ How to Run

Clone the repository:

git clone <repository-url>


Install dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn


Place the dataset data_banknote_authentication.txt in the same folder

Run the notebook/script:

python banknote_authentication.py


Outputs: Accuracy, Confusion Matrices, Classification Reports, ROC curves

📚 References

## References

[Banknote Authentication Dataset – UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)  
[Scikit-learn Documentation – KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)  
[Scikit-learn Documentation – SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)  
