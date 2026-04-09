# CS201L: Artificial Intelligence Laboratory
**Indian Institute of Technology, Dharwad**  
**Student:** Siddharth Shukla | **Roll No:** IS24BM039

---

This repository contains my lab submissions for the AI Laboratory course. Each lab covers a different concept in data analysis and machine learning, implemented in Python using Jupyter Notebooks.

---

## Labs Overview

### Lab 1 — Introduction to Python & Data Analysis
Getting started with NumPy, Pandas, and Matplotlib. Explored the Residential Housing dataset through basic data inspection, filtering, loops, and statistical operations like min, max, and sum.

### Lab 2 — Data Visualization & Statistics
Used Matplotlib and Seaborn to visualize datasets (Residential Housing & No-Show Appointments). Computed five-number summaries, quartiles, IQR, and generated scatter plots across key feature pairs.

### Lab 3 — Data Cleaning & Missing Value Handling
Worked on two datasets: Residential Housing and No-Show Appointments.
- Analyzed missing values per column and per row
- Applied imputation strategies: median filling, forward fill (`ffill`), backward fill (`bfill`), and linear interpolation
- Compared data statistics before and after imputation

### Lab 4 — Outlier Detection, Correlation & Dimensionality Reduction
Used the Energy Efficiency dataset.
- Detected and corrected outliers using IQR-based boxplot analysis
- Computed Pearson and Spearman correlation heatmaps between features
- Applied Min-Max and Z-score normalization, and explored PCA for dimensionality reduction

### Lab 5 — K-Nearest Neighbors (KNN) Classification
Used the Activity Detection dataset.
- Split data into Train (60%), Validation (20%), and Test (20%) with stratified sampling
- Applied Min-Max scaling, Standard scaling, and PCA transformations
- Trained and evaluated KNN classifiers across different data variants

### Lab 6 — Naive Bayes Classification
Used the Activity Detection dataset (pre-split from Lab 5).
- Implemented Gaussian Naive Bayes from scratch using class-wise means, covariances, and prior probabilities
- Compared with scikit-learn's `GaussianNB`
- Evaluated with confusion matrix, accuracy, and micro/macro precision, recall, F1-score

### Lab 7 — Logistic Regression & SVM
Ran experiments across four versions of the Activity Detection dataset: original, scaled, PCA (all components), and PCA (99% variance).
- Applied Logistic Regression and Support Vector Machine (SVM) with linear and RBF kernels
- Compared performance across all four data variants on both validation and test sets

### Lab 8 — Neural Networks with PyTorch
Used the Date Fruit dataset (898 samples, 34 features, 7 classes).
- Built single and two hidden layer neural networks using PyTorch (`tanh` activations, `Softmax` output)
- Trained with SGD and Cross-Entropy loss, with early stopping based on convergence threshold
- **Part A:** Raw data — compared 4 architectures, selected best per category
- **Part B:** Z-score normalized data — repeated the same experiments and compared results
- Plotted epoch vs. loss curves, saved trained models, and reported full metrics on test set

### Lab 9 — Linear & Polynomial Regression
Used the Energy Efficiency dataset.
- Fitted simple linear regression on a single feature to predict heating load
- Explored polynomial regression (degrees 2–11) to find best fit
- Evaluated using RMSE on train, validation, and test splits with visualizations

### Lab 10 — Multivariate & Ridge Regression
Used the Energy Efficiency dataset with all 9 input features.
- Extended polynomial regression to multiple features
- Applied Ridge regression to handle overfitting at higher polynomial degrees
- Compared RMSE curves across degrees and plotted actual vs. predicted values

### Lab 11 — Time Series Analysis & Forecasting
Used Apple (AAPL) historical stock price data.
- Analyzed autocorrelation (ACF) and lag-1 Pearson correlation of the closing price series
- Built an AutoRegressive (AR) model for time series forecasting
- Implemented an LSTM neural network (PyTorch) for sequence-based price prediction
- Compared AR and LSTM performance on train/test splits

---

## Tech Stack
- **Language:** Python 3
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, PyTorch, statsmodels
- **Environment:** Jupyter Notebook
