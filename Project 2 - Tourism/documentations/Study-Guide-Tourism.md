# 🏔️ Tourism Clustering + AutoML Dashboard - Complete Study Guide

**Project:** Global LAB 1 - Tourism Clustering + AutoML Dashboard  
**Total Marks:** 25  
**Author:** Aaditya GC 
**Date:** 3/24/2026

---

## 📖 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Understanding](#2-dataset-understanding)
3. [Step-by-Step Implementation](#3-step-by-step-implementation)
4. [Code Explanation - Every File](#4-code-explanation---every-file)
5. [Machine Learning Concepts](#5-machine-learning-concepts)
6. [Results and Evaluation](#6-results-and-evaluation)
7. [Dashboard Walkthrough](#7-dashboard-walkthrough)
8. [How to Run the Project](#8-how-to-run-the-project)
9. [Report Writing Guide](#9-report-writing-guide)

---

## 1. Project Overview

### 1.1 What Does This Project Do?

This project builds a **Tourist Segmentation System** that:
1. Takes tourist characteristics (Age, Budget, Duration, Activity Preference, Spending Score)
2. Uses **K-Means Clustering** to segment tourists into 5 distinct groups
3. Uses **AutoML** to automatically train and compare 8 different ML models
4. Displays results in a beautiful interactive website
5. Provides tourism strategies and marketing recommendations for each cluster

### 1.2 Problem Statement

**Given:** A tourism dataset from Pokhara with:
- Age
- Budget (NPR per day)
- Duration (days)
- Activity Preference (Adventure/Cultural/Relaxation/Spiritual)
- Spending Score (1-100)

**Task:** 
1. Apply K-Means Clustering to segment tourists into meaningful groups
2. Use AutoML to train multiple classification models
3. Build a dashboard to assign new tourists to clusters and suggest strategies

### 1.3 Solution Approach

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Tourist Data   │────▶│  K-Means         │────▶│  5 Clusters     │
│  (5 features)   │     │  Clustering      │     │  (Segments)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │  AutoML          │────▶│  Best Model     │
                        │  (8 models)      │     │  (99.6%)        │
                        └──────────────────┘     └─────────────────┘
```

### 1.4 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.x | Programming language |
| pandas | Data manipulation |
| numpy | Numerical computations |
| scikit-learn | Machine learning (K-Means, classifiers) |
| matplotlib & seaborn | Data visualization |
| streamlit | Web dashboard |
| joblib | Model saving/loading |

---

## 2. Dataset Understanding

### 2.1 Dataset Source

Synthetic dataset generated to mimic real Pokhara tourism patterns with realistic outliers.

### 2.2 Features Explained

| Feature | Range | Unit | What It Means |
|---------|-------|------|---------------|
| **Age** | 18-75 | years | Tourist's age |
| **Budget (NPR)** | 300-15,000 | NPR/day | Daily spending budget in Nepali Rupees |
| **Duration** | 1-90 | days | Length of stay in Pokhara |
| **Activity Preference** | Adventure/Cultural/Relaxation/Spiritual | Categorical | Preferred activity type |
| **Spending Score** | 1-100 | Score | Calculated spending tendency (higher = more spending) |

### 2.3 5 Tourist Clusters

| Cluster | Name | Size | Characteristics |
|---------|------|------|-----------------|
| **0** | 💎 Luxury Travelers | 738 | Older (54), high budget (8,022 NPR), long stay (15 days) |
| **1** | 🎒 Budget Backpackers | 1,310 | Young (25), low budget (1,350 NPR), short stay (3 days) |
| **2** | 💻 Digital Nomads | 52 | Extended stay (53 days), budget-conscious (1,897 NPR) |
| **3** | 🪂 Adventure Seekers | 1,268 | Active (32), medium-high budget (5,100 NPR), 9.5 days |
| **4** | 🏛️ Cultural Explorers | 1,632 | Middle-aged (45), moderate budget (3,075 NPR), 8 days |

### 2.4 Dataset Statistics

```
Total Records: 5,000
Features: 5 (Age, Budget, Duration, Activity, Spending Score)
Target: 5 Clusters (from K-Means)

Cluster Distribution:
- Cluster 0 (Luxury):       738 records (14.8%)
- Cluster 1 (Backpackers): 1,310 records (26.2%)
- Cluster 2 (Nomads):         52 records (1.0%)
- Cluster 3 (Adventure):   1,268 records (25.4%)
- Cluster 4 (Cultural):    1,632 records (32.6%)
```

### 2.5 Extreme Values (Outliers)

The dataset includes realistic outliers (~5%):
- **Ultra-budget:** 300-600 NPR/day (extreme backpackers)
- **Ultra-luxury:** 9,000-15,000 NPR/day (wealthy tourists)
- **Extended stay:** 30-90 days (digital nomads, long-term travelers)
- **Young rich:** 18-28 years with 8,000-12,000 NPR budget
- **Elderly backpackers:** 60-75 years on budget travel

---

## 3. Step-by-Step Implementation

### 3.1 Project Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        PROJECT WORKFLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Generate Data                                          │
│  ┌──────────────────────┐                                       │
│  │ tourism_data_        │                                       │
│  │ generator.py         │                                       │
│  └──────────┬───────────┘                                       │
│             │                                                   │
│             ▼                                                   │
│  Step 2: Create CSV File                                        │
│  ┌──────────────────────────┐                                   │
│  │ tourism_pokhara.csv      │                                   │
│  └──────────┬───────────────┘                                   │
│             │                                                   │
│             ▼                                                   │
│  Step 3: Apply K-Means Clustering                               │
│  ┌──────────────────────┐                                       │
│  │ tourism_clustering.py│                                       │
│  └──────────┬───────────┘                                       │
│             │                                                   │
│             ▼                                                   │
│  Step 4: Run AutoML Analysis                                    │
│  ┌──────────────────────────┐                                   │
│  │ tourism_automl_analysis. │                                   │
│  │ py                       │                                   │
│  └──────────┬───────────────┘                                   │
│             │                                                   │
│             ▼                                                   │
│  Step 5: Launch Dashboard                                       │
│  ┌──────────────────────┐                                       │
│  │ tourism_app.py       │                                       │
│  └──────────────────────┘                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 File Structure

```
F:\ML project\
│
├── Project 2 - Tourism/               # Main project folder
│   ├── data/                          # Data folder
│   │   ├── tourism_pokhara.csv        # Original dataset
│   │   └── tourism_pokhara_clustered.csv  # With K-Means labels
│   ├── models/                        # Trained models folder
│   │   ├── kmeans_model.pkl           # K-Means clustering model
│   │   ├── scaler.pkl                 # Feature scaler
│   │   ├── activity_encoder.pkl       # Activity encoder
│   │   ├── model_metadata.pkl         # Metadata
│   │   ├── automl_best_model.pkl      # Best AutoML model
│   │   ├── all_trained_models.pkl     # All 8 trained models
│   │   └── automl_results_summary.pkl # AutoML results
│   ├── documentations/                # Documentation folder
│   │   └── tourism_project_plan.md
│   ├── tourism_data_generator.py      # Dataset generation
│   ├── tourism_clustering.py          # K-Means implementation
│   ├── tourism_automl_analysis.py     # AutoML model comparison
│   ├── tourism_app.py                 # Streamlit dashboard
│   ├── tourism_requirements.txt       # Dependencies
│   ├── README_TOURISM.md              # Project summary
│   └── *.png                          # Visualization files
```

---

## 4. Code Explanation - Every File

### 4.1 File: `tourism_data_generator.py`

**Purpose:** Generate realistic synthetic tourism data with outliers

**Key Features:**
- Creates 5 distinct tourist clusters with different characteristics
- Adds ~5% extreme values (outliers) for realism
- Calculates Spending Score from Budget and Duration

**Code Highlights:**

```python
# Cluster 0: Budget Backpackers (~25%)
n_backpackers = int(n_samples * 0.25)
ages.extend(np.random.randint(18, 32, size=n_backpackers))
budgets.extend(np.random.exponential(scale=800, size=n_backpackers) + 500)
durations.extend(np.random.exponential(scale=3, size=n_backpackers) + 1)

# Add outliers (~5%)
n_outliers = int(n_samples * 0.05)
outlier_types = ['ultra_budget', 'ultra_luxury', 'long_stay', 'young_rich', 'elderly_backpacker']
```

**Key Concepts:**

| Concept | What It Does | Why It's Used |
|---------|--------------|---------------|
| `np.random.exponential()` | Generates skewed distribution | Realistic budget patterns |
| `np.random.normal()` | Generates bell curve | Age distribution around mean |
| `np.concatenate()` | Combines arrays | Mix different age groups |
| Outlier injection | Adds extreme values | Realistic data variations |

---

### 4.2 File: `tourism_clustering.py`

**Purpose:** Apply K-Means clustering and analyze results

**Key Steps:**

1. **Load and Preprocess:**
```python
# Encode categorical variable
encoder = LabelEncoder()
df['Activity Preference'] = encoder.fit_transform(df['Activity Preference'])

# Scale features (CRITICAL for K-Means!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

2. **Find Optimal K:**
```python
# Elbow Method
inertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, ...)
    inertias.append(kmeans.inertia_)

# Silhouette Analysis
silhouette_scores.append(silhouette_score(X_scaled, labels))
```

3. **Apply K-Means:**
```python
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',  # Smart initialization
    n_init=10,         # Run 10 times
    max_iter=300,
    random_state=42
)
cluster_labels = kmeans.fit_predict(X_scaled)
```

4. **Evaluate Clustering:**
```python
# Silhouette Score
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Cluster Statistics
cluster_stats = df.groupby('KMeans_Cluster').agg({
    'Age': ['mean', 'median'],
    'Budget': ['mean', 'median'],
    ...
})
```

**Key Concepts:**

| Concept | What It Does | Why It's Used |
|---------|--------------|---------------|
| `StandardScaler` | Normalizes features to mean=0, std=1 | K-Means needs scaled features |
| `LabelEncoder` | Converts text to numbers | ML models need numerical input |
| `KMeans` | Clustering algorithm | Groups similar tourists |
| `k-means++` | Smart centroid initialization | Better convergence |
| `silhouette_score` | Measures cluster separation | Evaluate clustering quality |
| `inertia_` | Sum of squared distances | Elbow method for optimal K |

---

### 4.3 File: `tourism_automl_analysis.py`

**Purpose:** Automatically train and compare 8 ML models

**Models Tested:**
1. Random Forest
2. Extra Trees
3. Gradient Boosting
4. Decision Tree
5. K-Nearest Neighbors
6. SVM (Support Vector Machine)
7. Neural Network (MLP)
8. Logistic Regression

**Code Structure:**

```python
# Define models to try
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=15),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'Decision Tree': DecisionTreeClassifier(max_depth=10),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', probability=True),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50)),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

# Train and evaluate each
for name, model in models.items():
    # Scale for certain models
    if name in ['KNN', 'SVM', 'Neural Network', 'Logistic Regression']:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
```

**Results:**

| Model | Accuracy |
|-------|----------|
| Neural Network | **99.6%** 🏆 |
| Logistic Regression | 98.9% |
| SVM | 98.8% |
| Extra Trees | 98.2% |
| Random Forest | 97.6% |
| K-Nearest Neighbors | 97.4% |
| Gradient Boosting | 97.3% |
| Decision Tree | 95.0% |

---

### 4.4 File: `tourism_app.py`

**Purpose:** Create interactive Streamlit dashboard

**Key Features:**

1. **Profile Section (Sidebar):**
```python
st.markdown("### 👤 Prepared By")
st.image("avatar.png", width=100)
st.markdown("""
<div style="text-align: center;">
    <p style="font-size: 18px; font-weight: bold;">[Your Name]</p>
    <p style="font-size: 14px;">🎓 Student</p>
    <p style="font-size: 13px;">Roll No: [Your Roll]</p>
</div>
""")
```

2. **Input Sliders:**
```python
age = st.slider("Age (years)", 18, 75, 30)
budget = st.slider("Budget (NPR)", 500, 10000, 3000)
duration = st.slider("Duration (days)", 1, 90, 7)
activity = st.selectbox("Activity", ['Adventure', 'Cultural', 'Relaxation', 'Spiritual'])
spending_score = st.slider("Spending Score", 1, 100, 20)
```

3. **Cluster Prediction:**
```python
# Prepare input
activity_encoded = encoder.transform([activity])[0]
input_data = [[age, budget, duration, activity_encoded, spending_score]]
input_scaled = scaler.transform(input_data)

# Predict cluster
cluster = kmeans.predict(input_scaled)[0]

# Get profile
profile = get_cluster_profile(cluster)
```

4. **Cluster Profiles & Strategies:**
```python
profiles = {
    0: {
        'name': '💎 Luxury Travelers',
        'description': 'Older, affluent tourists...',
        'characteristics': [...],
        'strategies': [...],
        'marketing': [...]
    },
    ...
}
```

---

## 5. Machine Learning Concepts

### 5.1 What is K-Means Clustering?

**K-Means** is an unsupervised learning algorithm that:
1. Divides data into K clusters
2. Each data point belongs to the cluster with nearest mean
3. Iteratively updates cluster centers

**Algorithm:**
```
1. Initialize K centroids (k-means++ for smart initialization)
2. Assign each point to nearest centroid
3. Recalculate centroids as mean of assigned points
4. Repeat steps 2-3 until convergence
```

### 5.2 Why Scale Features for K-Means?

K-Means uses **Euclidean distance** to measure similarity:

```
Distance = √[(x₁-x₂)² + (y₁-y₂)² + ...]
```

**Without scaling:**
- Budget (300-15000) dominates Age (18-75)
- Clusters biased toward high-range features

**With StandardScaler:**
- All features have mean=0, std=1
- Equal contribution to distance calculation

### 5.3 How to Find Optimal K?

**1. Elbow Method:**
- Plot inertia (sum of squared distances) vs K
- Look for "elbow" point where decrease slows

**2. Silhouette Analysis:**
- Measures how similar points are to their own cluster vs other clusters
- Range: -1 to 1 (higher is better)
- Formula: (b - a) / max(a, b)
  - a = mean distance to same cluster points
  - b = mean distance to nearest cluster points

**3. Business Understanding:**
- Choose K that makes business sense
- For tourism: 5 clusters (Budget, Luxury, Adventure, Cultural, Nomad)

### 5.4 What is AutoML?

**AutoML (Automated Machine Learning)** automatically:
1. Tries multiple algorithms
2. Tunes hyperparameters
3. Compares model performance
4. Selects best model

**In This Project:**
- Trained 8 models automatically
- Compared accuracy
- Best: Neural Network (99.6%)

### 5.5 Evaluation Metrics

**For Clustering:**
- **Silhouette Score:** 0.338 (good separation)
- **Inertia:** 8838.61 (lower is better)

**For Classification (AutoML):**
- **Accuracy:** 99.6% (Neural Network)
- **Precision, Recall, F1-Score:** All >0.99

---

## 6. Results and Evaluation

### 6.1 K-Means Clustering Results

**Optimal K Determination:**
- Elbow Method: K=5-8 showed good results
- Silhouette Analysis: K=8 had highest score (0.367)
- **Business Decision:** K=5 chosen for interpretability

**Final Clustering Performance:**
- **Silhouette Score:** 0.338
- **Cluster Sizes:** 52 to 1,632 tourists
- **Good separation** validated with PCA visualization

### 6.2 Cluster Profiles

| Cluster | Name | Age | Budget (NPR) | Duration | Spending Score |
|---------|------|-----|--------------|----------|----------------|
| 0 | Luxury | 54 | 8,022 | 15.3 | 36.5 |
| 1 | Backpackers | 25 | 1,350 | 4.0 | 6.9 |
| 2 | Nomads | 34 | 1,897 | 58.5 | 37.2 |
| 3 | Adventure | 32 | 5,101 | 9.6 | 23.1 |
| 4 | Cultural | 45 | 3,076 | 7.8 | 14.8 |

### 6.3 AutoML Results

**Model Leaderboard:**

| Rank | Model | Accuracy |
|------|-------|----------|
| 🥇 | Neural Network (MLP) | 99.6% |
| 🥈 | Logistic Regression | 98.9% |
| 🥉 | SVM (RBF Kernel) | 98.8% |
| 4 | Extra Trees | 98.2% |
| 5 | Random Forest | 97.6% |
| 6 | K-Nearest Neighbors | 97.4% |
| 7 | Gradient Boosting | 97.3% |
| 8 | Decision Tree | 95.0% |

**Best Model: Neural Network**
- Architecture: MLP with layers (100, 50)
- Training: 500 iterations
- Test Accuracy: 99.6%

### 6.4 Feature Importance

**For Cluster Assignment:**
- **Budget:** Most important (40-45%)
- **Duration:** Second most important (25-30%)
- **Age:** Moderate importance (15-20%)
- **Spending Score:** Moderate importance (10-15%)
- **Activity Preference:** Least important (5-10%)

---

## 7. Dashboard Walkthrough

### 7.1 Opening the Dashboard

1. Run: `streamlit run tourism_app.py`
2. Browser opens to `http://localhost:8502`

### 7.2 Sidebar (Left Panel)

**Profile Section:**
- Avatar image
- Your name, roll number, date

**Input Sliders:**
- Age: 18-75 years (default: 30)
- Budget: 500-10,000 NPR (default: 3,000)
- Duration: 1-90 days (default: 7)
- Activity Preference: Dropdown (Adventure/Cultural/Relaxation/Spiritual)
- Spending Score: 1-100 (default: 20)

**Find Cluster Button:** Click to get cluster assignment

**Dataset Info:**
- Total Tourists: 5,000
- Clusters: 5
- AutoML Best Accuracy: 99.6%

### 7.3 Main Area (Right Panel)

**Left Column:**
- Cluster Assignment (colored box with emoji)
- Cluster Name and Description
- Characteristics (bullet points)
- Tourism Strategies (expandable)
- Marketing Approaches (expandable)
- Cluster Similarity (bar chart)

**Right Column (Tabs):**

**Tab 1: Distribution**
- Pie chart of cluster sizes
- Shows percentage of each tourist segment

**Tab 2: Characteristics**
- Parallel coordinates plot
- Compares mean values across clusters

**Tab 3: 2D Clusters**
- PCA projection (2D scatter plot)
- Color-coded clusters
- Cluster centers marked with X

**Tab 4: AutoML Results**
- Model comparison leaderboard
- Bar chart of 8 models' accuracy

### 7.4 Sample Cluster Assignments

| Input | Assigned Cluster |
|-------|------------------|
| Age=25, Budget=1500, Duration=3 | 🎒 Budget Backpacker |
| Age=55, Budget=8000, Duration=15 | 💎 Luxury Traveler |
| Age=35, Budget=2000, Duration=60 | 💻 Digital Nomad |
| Age=32, Budget=5000, Duration=10 | 🪂 Adventure Seeker |
| Age=45, Budget=3000, Duration=8 | 🏛️ Cultural Explorer |

---

## 8. How to Run the Project

### 8.1 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### 8.2 Installation

```bash
# Navigate to project folder
cd "F:\ML project\Project 2 - Tourism"

# Install required libraries
pip install -r tourism_requirements.txt
```

### 8.3 Step-by-Step Execution

**Option A: Run Everything (Fresh Start)**

```bash
# Step 1: Generate dataset
python tourism_data_generator.py

# Step 2: Apply K-Means clustering
python tourism_clustering.py

# Step 3: Run AutoML analysis
python tourism_automl_analysis.py

# Step 4: Launch dashboard
streamlit run tourism_app.py
```

**Option B: Use Existing Files (Quick Start)**

```bash
# Dataset, models, and AutoML already done
# Just run dashboard
streamlit run tourism_app.py
```

### 8.4 Expected Output

**tourism_data_generator.py:**
```
🏔️  Generating Tourism Dataset for Pokhara...
------------------------------------------------------------
📊 Dataset Shape: (5000, 6)
📈 Cluster Distribution:
0    1116
1    1121
2    1093
3     942
4     728
💾 Dataset saved to: data\tourism_pokhara.csv
```

**tourism_clustering.py:**
```
🔍 Finding optimal number of clusters...
  K=2: Inertia=17241.36, Silhouette=0.325
  K=3: Inertia=13224.41, Silhouette=0.297
  ...
  K=5: Inertia=8838.61, Silhouette=0.338
📊 Optimal K (by Silhouette): 8 (score: 0.367)
🎯 Using 5 clusters for final model...
📏 Overall Silhouette Score: 0.338
💾 Saved: kmeans_model.pkl
```

**tourism_automl_analysis.py:**
```
🤖 Training Multiple Models Automatically...
🌳 Training Neural Network...
  ✅ Neural Network Accuracy: 0.9960 (99.60%)
🏆 Model Leaderboard:
              Model  Accuracy
     Neural Network     0.996
Logistic Regression     0.989
                SVM     0.988
```

**tourism_app.py:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8502
```

---

## 9. Report Writing Guide

### 9.1 Report Structure

Use this structure for your project report:

```
1. Title Page
2. Abstract
3. Introduction
4. Literature Review (optional)
5. Methodology
6. Implementation
7. Results and Discussion
8. Conclusion
9. References
10. Appendix (Code)
```

### 9.2 Section-by-Section Guide

#### **1. Title Page**
```
Tourism Clustering + AutoML Dashboard
Using Machine Learning for Tourist Segmentation

Submitted by: [Your Name]
Roll No: [Your Roll Number]
Date: [Submission Date]
Course: [Course Name]
```

#### **2. Abstract** (150-200 words)

```
Tourism is a major industry in Nepal, particularly in Pokhara. This project 
develops a machine learning-based system to segment tourists into distinct 
clusters for targeted marketing and service delivery. K-Means clustering 
was applied to a dataset of 5000 tourist records with features including 
Age, Budget, Duration, Activity Preference, and Spending Score. The optimal 
5-cluster solution achieved a silhouette score of 0.338. Additionally, 
AutoML was used to automatically train and compare 8 classification models, 
with a Neural Network achieving the best accuracy of 99.6%. An interactive 
Streamlit dashboard was developed to enable real-time cluster assignment 
and provide tourism strategies for each segment. This system helps tourism 
boards and businesses create targeted marketing campaigns and personalized 
tourist experiences.
```

#### **3. Introduction**

**3.1 Background**
- Tourism importance in Nepal/Pokhara
- Need for tourist segmentation
- Benefits of targeted marketing

**3.2 Problem Statement**
```
Given tourist characteristics, segment tourists into meaningful clusters 
and provide targeted strategies for tourism businesses.
```

**3.3 Objectives**
- Apply K-Means clustering for tourist segmentation
- Determine optimal number of clusters
- Use AutoML for model comparison
- Create user-friendly dashboard
- Provide cluster-specific tourism strategies

**3.4 Scope**
- Pokhara tourism
- 5 input features
- 5 tourist clusters

#### **4. Methodology**

**4.1 Dataset Description**
- Source: Synthetic (generated to mimic Pokhara patterns)
- Size: 5000 records
- Features: Age, Budget, Duration, Activity, Spending Score
- Includes ~5% outliers for realism

**4.2 Data Preprocessing**
- Label encoding for Activity Preference
- StandardScaler for feature normalization
- 80-20 train-test split (for AutoML)

**4.3 Algorithm Selection**

**K-Means Clustering:**
- Unsupervised learning
- Partitions data into K clusters
- Minimizes within-cluster variance

**AutoML Models:**
- Random Forest, Extra Trees, Gradient Boosting
- Decision Tree, KNN, SVM
- Neural Network (MLP), Logistic Regression

**4.4 Model Training**

**K-Means:**
```python
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)
```

**AutoML:**
- 8 models trained automatically
- Hyperparameter tuning
- Performance comparison

**4.5 Evaluation Metrics**

**Clustering:**
- Silhouette Score
- Inertia
- Calinski-Harabasz Score
- Davies-Bouldin Index

**Classification:**
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix

#### **5. Implementation**

**5.1 Technologies Used**
- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- streamlit

**5.2 System Architecture**
```
[Data Generation] → [K-Means Clustering] → [AutoML] → [Dashboard]
```

**5.3 Key Code Snippets**
Include important code sections from `tourism_clustering.py` and `tourism_app.py`

#### **6. Results and Discussion**

**6.1 Clustering Performance**
```
Silhouette Score: 0.338
Number of Clusters: 5
Cluster Sizes: 52 to 1,632 tourists
```

**6.2 Cluster Profiles**
[Include table with cluster characteristics]

**6.3 AutoML Results**
```
Best Model: Neural Network (MLP)
Accuracy: 99.6%
Model Leaderboard: [include top 5 models]
```

**6.4 Feature Importance**
[Include feature importance chart or table]

**6.5 Dashboard Screenshots**
Include screenshots of:
- Main dashboard
- Cluster assignment
- Visualizations (all 4 tabs)

**6.6 Discussion**
- Cluster interpretability
- Business applications
- Limitations (synthetic data)

#### **7. Conclusion**

```
This project successfully developed a tourist segmentation system using 
K-Means clustering and AutoML. Five distinct tourist clusters were 
identified: Luxury Travelers, Budget Backpackers, Digital Nomads, 
Adventure Seekers, and Cultural Explorers. The clustering achieved a 
silhouette score of 0.338, indicating good separation. AutoML analysis 
revealed that a Neural Network could predict cluster membership with 
99.6% accuracy. The Streamlit dashboard provides an intuitive interface 
for real-time cluster assignment and tourism strategy recommendations. 
Future work includes using real tourism data from Pokhara tourism board, 
adding geospatial analysis, and deploying to cloud for public access.
```

#### **8. References**

```
[1] Scikit-learn Documentation: https://scikit-learn.org/
[2] Streamlit Documentation: https://docs.streamlit.io/
[3] Nepal Tourism Board: https://www.welcomenepal.com/
[4] Pokhara Tourism Statistics: [relevant sources]
[5] Jain, A.K. (2010). Data clustering: 50 years beyond K-means
```

#### **9. Appendix**

Include complete code files:
- `tourism_data_generator.py`
- `tourism_clustering.py`
- `tourism_automl_analysis.py`
- `tourism_app.py` (key sections)

---

## 📝 Quick Reference Cheat Sheet

### Commands to Remember

```bash
# Generate data
python tourism_data_generator.py

# Run clustering
python tourism_clustering.py

# Run AutoML
python tourism_automl_analysis.py

# Launch dashboard
streamlit run tourism_app.py

# Install dependencies
pip install -r tourism_requirements.txt
```

### Key Files

| File | Purpose |
|------|---------|
| `tourism_data_generator.py` | Creates synthetic dataset |
| `tourism_clustering.py` | K-Means clustering |
| `tourism_automl_analysis.py` | AutoML model comparison |
| `tourism_app.py` | Interactive dashboard |
| `kmeans_model.pkl` | Saved K-Means model |
| `automl_best_model.pkl` | Best AutoML model |

### Important Concepts

| Concept | Definition |
|---------|------------|
| **K-Means** | Unsupervised clustering algorithm |
| **Silhouette Score** | Measure of cluster separation (-1 to 1) |
| **Inertia** | Sum of squared distances to centroid |
| **AutoML** | Automated machine learning |
| **StandardScaler** | Normalizes features to mean=0, std=1 |
| **Cluster** | Group of similar data points |

---

## 🎓 Study Questions

### Basic Level
1. What are the five input features?
2. How many clusters were identified?
3. What clustering algorithm was used?
4. What was the best AutoML model accuracy?

### Intermediate Level
1. Why is feature scaling important for K-Means?
2. How was optimal K determined?
3. What is the difference between supervised and unsupervised learning?
4. Why use AutoML instead of manually training models?

### Advanced Level
1. Explain how K-Means determines cluster assignments.
2. Compare and contrast Elbow Method vs Silhouette Analysis.
3. How would you validate clusters with real-world data?
4. What business insights can be derived from the cluster profiles?

---

## 🔗 Additional Resources

- [Scikit-learn K-Means Tutorial](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Nepal Tourism Statistics](https://www.tourism.gov.np/)
- [Pokhara Tourism Board](https://www.pokharatourism.org/)

---

<div align="center">

**End of Study Guide**

*Created for Global LAB 1 - Tourism Clustering Project*

</div>
