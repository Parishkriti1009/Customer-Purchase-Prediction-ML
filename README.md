# Bank Marketing Customer Term Deposit Prediction

## 📌 Project Overview
This repository contains an end-to-end predictive data science pipeline built to determine whether a banking client will subscribe to a term deposit product. Utilizing historical direct telephonic marketing campaign data, the system implements predictive modeling to optimize targeting strategies, reduce campaign outreach overhead, and maximize financial conversion rates.

## 📈 Key Results
* **Classification Accuracy:** **91.54%** achieved on the validation split.
* **Model Pipeline:** Built and optimized using a Supervised `DecisionTreeClassifier`.

## 📊 Dataset Profile
The project utilizes the **UCI Bank Marketing Dataset** to analyze consumer behavioral features and macro-economic factors:
* **Dataset Scale:** 41,188 consumer records.
* **Feature Space:** 21 structural attributes spanning customer demographics (age, job, marital status), loan histories, campaign contact frequencies, and economic indicators.
* **Target Vector:** `y` (Binary variable: Has the client subscribed to a term deposit?).

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Engineering & EDA:** NumPy, Pandas
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning Engine:** Scikit-Learn (`DecisionTreeClassifier`, `LabelEncoder`, `train_test_split`)

## ⚙️ Architecture & Data Pipeline
1. **Exploratory Data Analysis (EDA):** Evaluated feature distributions, tracked target class distributions, and visualized correlative trends between demographic backgrounds and product adoption.
2. **Data Preprocessing & Encoding:** Handled missing data entries and transformed non-numeric attributes using automated `LabelEncoder` pipelines to generate algorithm-ready numeric metrics.
3. **Model Training & Validation:** Partitioned data arrays into stratified training and testing subsets to protect against data leakage and evaluated the tree classifier configurations.
4. **Performance Evaluation:** Quantified classification patterns via confusion matrices, isolating precision and recall thresholds.

## 🚀 Getting Started

### Prerequisites
Install the required data science dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
