# 🤖 Artificial Intelligence Lab — CS201L (2026)

**Student:** Siddharth Shukla | **Roll No:** IS24BM039  
**Course:** Artificial Intelligence Lab (CS201L)  
**Institution:** [IIT DHARWAD]  
**Academic Year:** 2025–2026

---

## 📖 About This Repository

This repository contains all lab submissions, practice notebooks, and assignment solutions for the **Artificial Intelligence Lab (CS201L)**. The labs are structured progressively — starting from Python fundamentals and data analysis, moving through machine learning algorithms, and culminating in deep learning and unsupervised clustering techniques. Each lab folder contains the student notebook (`.ipynb`), problem statement PDFs, and relevant datasets.

---

## 📂 Repository Structure

```
AI_lab/
├── lab_1/       → Python & NumPy/Pandas Fundamentals
├── lab_2/       → Data Visualization & Statistics
├── lab_3/       → Data Cleaning & Missing Value Handling
├── lab_4/       → Outlier Detection, Correlation & PCA
├── lab_5/       → K-Nearest Neighbors (KNN) Classification
├── lab_6/       → NumPy Operations & Linear Algebra
├── lab_7/       → SVM & Logistic Regression (Classification)
├── lab_8/       → Neural Networks (Assignment)
├── lab_9/       → Linear & Polynomial Regression
├── lab 10/      → Ridge Regression & Regularization
├── lab 11/      → Time Series Analysis & LSTM (Stock Prediction)
├── lab 12/      → K-Means & K-Medoids Clustering
└── lab 13/      → Hierarchical Clustering & DBSCAN
```

---

## 🧪 Lab Details

---

### 🔬 Lab 1 — Introduction to Python & Data Analysis with NumPy and Pandas

**File:** `lab_1/siddharth shukla _ is2bm039 _ problm statement 1 and 2.ipynb`

This lab forms the foundation of the entire AI course by introducing core Python data science libraries. The dataset used is **Residential Housing Data**, which contains real estate attributes such as lot area, number of bedrooms, and sale prices. Students begin by loading CSV data using Pandas and perform exploratory queries using loops, conditional logic, and basic data structures.

Key concepts covered include:
- Loading and exploring real-world datasets with `pd.read_csv()`
- Using `for` and `while` loops to iterate over DataFrame columns
- Applying conditional filters (e.g., counting houses with LotArea > 2000 sq ft)
- Computing cumulative statistics like total sale price of the first 15 entries
- Working with Python data structures — lists, tuples, dictionaries — in the context of real data
- Zipping multiple columns together for parallel processing
- Building custom dictionaries to map LotArea → SalePrice relationships
- Using NumPy for vectorized operations on housing attributes
- Accessing and slicing DataFrame rows and columns for targeted analysis

This lab builds the fundamental coding intuition needed for all subsequent labs. Students learn how to think in terms of data pipelines and programmatic analysis rather than spreadsheet-style manual computation.

---

### 🔬 Lab 2 — Data Visualization & Statistics

**File:** `lab_2/is24bm039_ps1.ipynb`, `lab_2/is24bm039_ps2.ipynb`

This lab dives into statistical exploration and visual analysis using the **No-Show Appointments dataset** — a real-world Brazilian hospital dataset with over 100,000 medical appointment records. Students extract meaningful patterns from patient data including age, scholarship status, SMS reminders, and whether patients showed up.

Key concepts covered include:
- Computing descriptive statistics: mean, median, standard deviation, variance using `.describe()`
- Calculating Quartiles (Q1, Q2, Q3) and Interquartile Range (IQR) manually and with Pandas
- Detecting statistical outliers using the 1.5×IQR rule
- Creating scatter plot matrices to visualize pairwise relationships between numeric features
- Building histograms with overlaid normal distribution curves using Matplotlib
- Using Seaborn for visually appealing distribution and count plots
- Analyzing the distribution of patient age and its relationship to appointment no-shows
- Computing the percentage of data outside the IQR bounds
- Producing publication-quality multi-panel figures for all numeric columns

This lab trains students to look at data statistically before applying any machine learning model — a critical step in any real-world AI pipeline.

---

### 🔬 Lab 3 — Data Cleaning & Handling Missing Values

**File:** `lab_3/is24bm039_lab3_p1_1.ipynb`, `is24bm039_lab3_p1_2.ipynb`, `is24bm039_lab3_p2_1.ipynb`, `is24bm039_lab3_p2_2.ipynb`

This lab addresses one of the most critical real-world challenges in data science: **missing data**. Using an updated version of the No-Show Appointments dataset with deliberately introduced missing values, students implement and compare multiple strategies for handling null entries.

Key concepts covered include:
- Detecting missing values per column using `.isna().sum()` and computing total null count
- Understanding when to impute vs. drop missing data
- **Median Imputation:** Replacing NaN values in numeric columns with the column median — robust against outliers
- **Mode Imputation:** Filling categorical columns with the most frequent value
- **Forward Fill (ffill):** Propagating the last valid observation forward to fill gaps — useful for time-series-like data
- **Backward Fill (bfill):** Filling missing values using the next valid observation
- **Linear Interpolation:** Estimating missing values by fitting a straight line between surrounding known values
- Comparing pre- and post-imputation descriptive statistics (mean, std, median, mode) to assess the impact of each method
- Verifying zero remaining null values after each cleaning strategy

Students learn that no single method is universally best — the choice depends on data type, distribution, and domain context.

---

### 🔬 Lab 4 — Outlier Detection, Correlation Analysis & PCA

**File:** `lab_4/is24bm039_lab4_p1.ipynb`

This lab uses the **Energy Efficiency dataset** (UCI Machine Learning Repository) — a dataset of buildings with architectural parameters (wall area, roof area, glazing area, etc.) and heating/cooling load targets. The objective is to prepare the dataset properly before any model training through rigorous preprocessing.

Key concepts covered include:
- **Outlier Detection using IQR:** Identifying extreme values in each feature column using the Q1−1.5×IQR and Q3+1.5×IQR bounds
- Visualizing outliers using **boxplots** before and after correction
- **Outlier Correction:** Replacing detected outliers with the column median to preserve data volume while reducing noise
- Computing **Pearson and Spearman correlation matrices** between all features
- Visualizing correlation matrices as annotated heatmaps using Seaborn
- Identifying highly correlated feature pairs that may cause multicollinearity
- Performing correlation analysis with respect to targets Y1 (Heating Load) and Y2 (Cooling Load)
- Applying **Min-Max Scaling** and **Standard Scaling** to normalize features
- Implementing **Principal Component Analysis (PCA)** to reduce dimensionality while retaining variance
- Plotting bar charts of feature correlations to visually rank predictor importance

This lab teaches the complete preprocessing pipeline that must precede any regression or classification task.

---

### 🔬 Lab 5 — K-Nearest Neighbors (KNN) Classification

**File:** `lab_5/is24bm039_lab5_p1.ipynb`

This lab applies **K-Nearest Neighbors**, one of the most intuitive machine learning algorithms, on the **Activity Detection dataset** — a sensor-based human activity recognition dataset where the target variable is the physical activity being performed (walking, running, sitting, etc.). The dataset has high dimensionality with many numeric sensor features.

Key concepts covered include:
- Performing a stratified **three-way split** of the dataset: 60% Train / 20% Validation / 20% Test — preserving class proportions across all splits
- Saving train/validation/test splits as separate CSVs for reproducibility
- Applying **MinMaxScaler** and **StandardScaler** and comparing their effect on KNN performance
- Running **PCA** for dimensionality reduction and evaluating how many components preserve 95% of variance
- Training KNN classifiers for multiple values of K (neighbors) and comparing performance
- Evaluating models using **accuracy, precision, recall, and F1-score** (both micro and macro averages)
- Generating and interpreting **confusion matrices** to identify which activities are most often confused
- Plotting **accuracy vs. K** curves to identify the optimal number of neighbors
- Investigating the effect of distance metrics (Euclidean, Manhattan) on model performance
- Understanding the **curse of dimensionality** and why scaling and PCA matter for KNN

This lab provides hands-on experience with the complete supervised ML workflow: split → preprocess → train → evaluate → tune.

---

### 🔬 Lab 6 — NumPy Operations & Linear Algebra Foundations

**File:** `lab_6/NumPy_Basic_Operations_Practice.ipynb`, `lab_6/is24bm039_lab6_p1.ipynb`

This lab focuses on the mathematical backbone of machine learning — **linear algebra operations** — implemented using NumPy. Students work through progressively complex array operations that underpin all ML algorithms from logistic regression to deep neural networks.

Key concepts covered include:
- Scalar, vector, and matrix **element-wise multiplication and division**
- Understanding **broadcasting** rules in NumPy — how arrays of different shapes are combined automatically
- Using `np.argmax()` on both 1D arrays and 2D arrays (row-wise and column-wise) — mimicking what happens inside a softmax classifier
- **Reshaping arrays:** converting 1D vectors to row vectors (1, n) or column vectors (n, 1) using `reshape()`
- **Transposing matrices** using `.T` and understanding how this changes dot product orientation
- Matrix multiplication with `np.dot()` and the `@` operator — the core operation of every neural network layer
- Understanding **posteriors** and how argmax maps a probability distribution to a class prediction
- Practicing dimension manipulation with `np.expand_dims()`, `np.squeeze()`, and `reshape()`
- Stack and concatenate operations with `np.vstack()` and `np.hstack()`

This lab reinforces the insight that all deep learning is, at its core, arrays multiplying other arrays — and that mastering these operations is essential for debugging any model.

---

### 🔬 Lab 7 — Support Vector Machines & Logistic Regression

**File:** `lab_7/is24bm039_lab7_1.ipynb` through `is24bm039_lab7_4.ipynb`

This lab explores two powerful classification algorithms on the pre-processed **Activity Detection dataset** (scaled features from Labs 4 and 5). Students implement and benchmark **Logistic Regression** and **Support Vector Machines** with different kernels, gaining a comparative understanding of linear vs. non-linear decision boundaries.

Key concepts covered include:
- Training a **Logistic Regression** model with `max_iter=1000` and extracting learned weights (coefficients)
- Predicting on both validation and test sets and evaluating with full classification metrics
- Interpreting **confusion matrices** for multi-class activity classification
- Training a **Linear SVM** (`kernel='linear'`) and understanding the maximum-margin hyperplane concept
- Training an **RBF SVM** (`kernel='rbf'`) and understanding how it implicitly maps data to higher-dimensional spaces via the kernel trick
- Training a **Polynomial SVM** with varying degrees and observing potential overfitting behavior
- Comparing accuracy, precision, recall, and F1-score across all four models side by side
- Plotting confusion matrices as heatmaps for intuitive visual interpretation
- Tuning the **C hyperparameter** (regularization) in SVM and studying the bias-variance tradeoff
- Understanding when Logistic Regression is preferred over SVM (speed, interpretability) and vice versa (high-dimensional, non-linear data)

Each of the 4 notebooks corresponds to a distinct model variant, enabling direct controlled comparison.

---

### 🔬 Lab 8 — Neural Networks (Assignment)

**Files:** `lab_8/Lab08_Assignment.pdf`, `lab_8/Lab08_Practice (1).pdf`

This lab introduces **Artificial Neural Networks** through structured assignment and practice problems. The lab PDFs contain detailed problem statements covering feedforward network design, backpropagation theory, and practical implementation guidelines.

Key concepts covered in the assignment include:
- Architecture design: choosing the number of layers, neurons per layer, and activation functions (ReLU, Sigmoid, Tanh)
- Understanding the **forward pass** — computing activations layer by layer from input to output
- Deriving and implementing the **backpropagation** algorithm — computing gradients via the chain rule
- Choosing appropriate **loss functions** for classification (cross-entropy) vs. regression (MSE)
- Applying **gradient descent** and its variants (SGD, Adam, RMSProp) with learning rate tuning
- Regularization strategies: **L1/L2 weight decay** and **Dropout** to prevent overfitting
- Understanding batch size effects on convergence speed and generalization performance
- The **vanishing and exploding gradient problems** in deep networks and how to mitigate them
- Role of **batch normalization** in stabilizing and accelerating training
- Training a simple fully connected network on a structured dataset and evaluating on held-out test data

---

### 🔬 Lab 9 — Linear & Polynomial Regression

**File:** `lab_9/lab9.ipynb`

This lab introduces **supervised regression** using the **Energy Efficiency dataset**, beginning with the simplest possible model (single-feature linear regression) and systematically building up to high-degree polynomial models. The goal is to intuitively understand model complexity, overfitting, and underfitting — before applying regularization in the next lab.

Key concepts covered include:
- Building a **Simple Linear Regression** model using only feature X1 (Relative Compactness) to predict heating load Y
- Computing **Root Mean Squared Error (RMSE)** separately on training, validation, and test sets
- Plotting the regression line overlaid on scatter plots to visually assess fit quality
- Constructing **Polynomial Features** of degrees 2 through 11 using `PolynomialFeatures`
- Training a separate `LinearRegression` model for each degree and recording train/val RMSE
- Plotting RMSE vs. polynomial degree curves — identifying the "elbow" where validation error starts increasing (overfitting begins)
- Understanding the concept of **model complexity**: why higher-degree polynomials fit training data perfectly but fail to generalize
- Visualizing predicted vs. actual heating loads as scatter plots for each degree
- Selecting the best polynomial degree using validation performance (not training performance)

This lab is the conceptual bridge between pure statistical analysis (Lab 2) and regularized machine learning (Lab 10).

---

### 🔬 Lab 10 — Ridge Regression & Regularization

**File:** `lab 10/is24bm039_lab10_p1.ipynb`

This lab directly extends Lab 9 by introducing **regularization** to combat the overfitting seen in high-degree polynomial regression. Using the same Energy Efficiency dataset but now with all 9 features (X1–X9), students compare unregularized polynomial regression with **Ridge Regression** (L2 regularization).

Key concepts covered include:
- Multi-feature polynomial regression: transforming all 9 input features using `PolynomialFeatures` and observing the explosion in feature count
- Sweeping polynomial degrees 1–6 and computing train/val RMSE for each — selecting the best degree based on validation performance
- Visualizing the **RMSE vs. Degree curve** with separate train and validation lines to clearly spot the overfitting region
- Implementing **Ridge Regression** using `Ridge(alpha=...)` and understanding the L2 penalty term
- Sweeping multiple alpha values and observing how larger alpha shrinks weight magnitudes and reduces variance
- Comparing **Unregularized vs. Ridge** side-by-side using grouped bar charts of train and validation RMSE
- Plotting **predicted vs. actual heating load** scatter plots with an ideal-fit reference diagonal
- Understanding the mathematical intuition: Ridge adds λ‖w‖² to the MSE loss, discouraging large weights
- Selecting the best alpha empirically via validation RMSE
- Discussing why Ridge is preferred over plain polynomial regression when features are highly correlated (multicollinearity)

By the end of this lab, students have a complete framework for the bias-variance tradeoff and can select appropriate regularization strength empirically.

---

### 🔬 Lab 11 — Time Series Analysis & LSTM (Stock Price Prediction)

**File:** `lab 11/is24bm039_lab11_1.ipynb`

This is one of the most advanced labs, applying **time series analysis** and **deep learning (LSTM)** to real historical **Apple Inc. (AAPL) stock price data** spanning many years. The lab bridges classical statistical time series methods with modern sequence modeling using recurrent neural networks.

Key concepts covered include:
- Loading and indexing time series data by date using `pd.to_datetime()` and `.set_index()`
- Visualizing AAPL closing price trends over time — observing exponential growth from 2019 onwards and peaks approaching $250+
- Computing **Pearson Correlation** between the closing price and its lag-1 value — confirming extremely high autocorrelation (~0.9998)
- Plotting **lag scatter plots** to visualize the near-perfect linear dependence between consecutive days
- Generating the **Autocorrelation Function (ACF)** plot up to 600 lags to understand the long-range memory of the series
- Splitting data into 90% train / 10% test while preserving temporal order (absolutely no random shuffling)
- Implementing a classical **Autoregressive (AR) model** manually — predicting the next value from the last p observed values using `statsmodels`
- Building an **LSTM neural network** using PyTorch (`torch.nn.LSTM`) for sequence-to-value modeling
- Creating sliding window sequences for LSTM training — converting the series into (input_window → next_value) input-output pairs
- Training the LSTM with Adam optimizer and MSE loss, monitoring loss curves across epochs
- Applying **StandardScaler** before training and inverse-transforming predictions back to the original price domain
- Comparing AR model vs. LSTM on RMSE and overlaid time series prediction plots

This lab is a complete deep learning mini-project, showcasing the power of recurrent networks for sequential financial data.

---

### 🔬 Lab 12 — K-Means & K-Medoids Clustering

**File:** `lab 12/is24bm039_lab12.ipynb`

This lab introduces **unsupervised learning** through clustering. The dataset is **t-SNE reduced 2D features of 6 programming languages** — audio or code samples projected to 2D space using t-SNE for visualization. The goal is to cluster these samples without using labels, and then evaluate how well the resulting clusters match the true language labels.

Key concepts covered include:
- Loading and visualizing t-SNE 2D feature data with per-language color-coded scatter plots to understand natural groupings
- Understanding cluster evaluation metrics: **Purity, Normalized Mutual Information (NMI), and Silhouette Score**
- Implementing a custom **Purity Score** function from scratch using `Counter` — the fraction of correctly assigned points in each cluster
- Running **K-Means clustering** for K = 2 through 10 and computing all three metrics for each K
- Plotting **Inertia (Within-Cluster Sum of Squares)** vs. K — the **Elbow Method** for choosing optimal K
- Generating a 3×3 grid of cluster scatter plots (one per K value) with centroids marked in black
- Implementing **K-Medoids clustering** using `KMedoids` from `sklearn_extra` — a more robust alternative that uses actual data points (not means) as cluster centers
- Comparing K-Means vs. K-Medoids at K=6 on Purity, NMI, and Silhouette Score
- Discussing why K-Medoids is less sensitive to outliers compared to K-Means

The lab demonstrates that unsupervised clustering can recover meaningful structure (language groupings) without ever seeing ground truth labels during training.

---

### 🔬 Lab 13 — Hierarchical Clustering & DBSCAN

**File:** `lab 13/is24bm039_lab13.ipynb`

The final lab extends clustering to **hierarchical** and **density-based** methods on the same 6-language t-SNE dataset from Lab 12. This allows direct comparison of multiple unsupervised clustering paradigms on identical data, giving students a comprehensive view of the clustering algorithm landscape.

Key concepts covered include:
- Encoding string language labels to integers using `LabelEncoder` for metric computation
- Implementing a custom `purity_score()` function that correctly handles noise labels (−1) produced by DBSCAN
- Running **Agglomerative Hierarchical Clustering** with 4 different linkage methods: `ward`, `complete`, `average`, and `single`
- Comparing all 4 linkage strategies in a formatted summary table of Purity, NMI, and Silhouette — identifying `ward` as typically the best-performing method
- Building full **Dendrograms** using `scipy.cluster.hierarchy.dendrogram` with multiple distance thresholds (50, 100, 60% of max distance)
- Counting how many clusters form at each threshold level and annotating the threshold cuts on dendrogram plots
- Implementing **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) — an algorithm that finds arbitrarily shaped clusters and identifies noise points as label −1, requiring no pre-specified K
- Tuning DBSCAN `epsilon` (neighborhood radius) and `min_samples` parameters and observing their effect on cluster count and noise ratio
- Generating side-by-side comparison plots of hierarchical vs. DBSCAN cluster assignments on the same 2D data
- Building a final comparison table across all methods: K-Means vs. K-Medoids vs. Agglomerative (best linkage) vs. DBSCAN

This lab provides a holistic view of the unsupervised learning landscape and equips students to choose the right clustering algorithm based on data geometry, cluster shape assumptions, and sensitivity to noise.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `numpy` | Numerical computation, array operations |
| `pandas` | Data loading, manipulation, aggregation |
| `matplotlib` | Plotting and visualization |
| `seaborn` | Statistical visualization |
| `scikit-learn` | ML algorithms, preprocessing, metrics |
| `sklearn_extra` | K-Medoids clustering |
| `statsmodels` | AutoRegression (AR) model, ACF plots |
| `torch` (PyTorch) | LSTM neural network for time series |
| `scipy` | Hierarchical clustering, dendrogram |

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/siddharth277/Ai-lab.git]
cd Ai-lab
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-learn-extra statsmodels torch scipy notebook
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Navigate to any lab folder and open the `.ipynb` file.

---

## 🚀 How to Push This to GitHub (Step-by-Step)

### ⚠️ Important Note
Your `AI_lab` folder **already has a `.git` folder inside it** — meaning it was previously initialized as a git repository. So you do NOT need to run `git init`. You only need to connect it to your GitHub account and push.

### Step 1 — Create a GitHub Repository
1. Go to [github.com](https://github.com) → click **+** → **New repository**
2. Name it `AI_lab`
3. Choose **Public** or **Private**
4. **Do NOT** initialize with a README (you already have one)
5. Click **Create repository** and copy the repo URL (e.g. `[https://github.com/siddharth277/Ai-lab.git]`)

### Step 2 — Open Terminal Inside the AI_lab Folder

```bash
# Check current remotes
git remote -v
```

### Step 3 — Connect to Your GitHub Repo

```bash
# If no remote exists:
git remote add origin https://github.com/siddharth277/Ai-lab.git

# If a remote already exists but is wrong:
git remote set-url origin https://github.com/siddharth277/Ai-lab.git
```

### Step 4 — Copy This README into the Folder

Place the downloaded `README.md` file inside your `AI_lab/` folder.

### Step 5 — Stage, Commit & Push

```bash
# Stage all files
git add .

# Commit
git commit -m "Add all AI lab submissions with detailed README"

# Set branch and push
git branch -M main
git push -u origin main
```

> GitHub will ask for your **username** and a **Personal Access Token** as password.  
> Generate one at: GitHub → Settings → Developer Settings → Personal Access Tokens → **Generate new token (classic)** → check `repo` → copy the token and paste it as your password.

### Step 6 — Verify ✅
Open `https://github.com/siddharth277/Ai-lab.git` in your browser — all labs and the README will render on the homepage.

---

## 📄 License

This repository is for academic and educational purposes only.  
© 2026 Siddharth Shukla | IS24BM039
