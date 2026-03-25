# 🏔️ Tourism Clustering + AutoML Dashboard - Project Plan

**Total Marks: 25**  
**Location: Pokhara Tourism Data**

---

## 📋 Project Overview

Build a machine learning pipeline that:
1. Preprocesses tourism data from Pokhara
2. Applies K-Means Clustering to segment tourists
3. Uses MLJAR AutoML to automatically train and compare models
4. Deploys an interactive Streamlit dashboard for cluster visualization
5. Interprets results and suggests tourism strategies

---

## 🗂️ Phase 1: Dataset Creation & Preprocessing (5 marks)

### 1.1 Synthetic Dataset Generation
**File:** `data_generator.py`

Create a realistic synthetic dataset with the following features:

| Feature | Type | Range/Values | Description |
|---------|------|--------------|-------------|
| Age | Numeric | 18-70 years | Tourist age |
| Budget | Numeric | 500-10000 NPR | Daily budget in Nepali Rupees |
| Duration | Numeric | 1-30 days | Length of stay |
| Activity Preference | Categorical | Adventure/Cultural/Relaxation/Spiritual | Preferred activity type |
| Spending Score | Numeric | 1-100 | Calculated from budget and duration |

**Cluster Patterns (for realistic segmentation):**
- **Cluster 0 (Budget Backpackers):** Young (18-30), Low budget (500-2000), Short stay (1-5 days)
- **Cluster 1 (Cultural Explorers):** Middle-aged (35-55), Medium budget (2000-5000), Medium stay (5-10 days)
- **Cluster 2 (Luxury Travelers):** Older (45-70), High budget (5000-10000), Long stay (10-30 days)
- **Cluster 3 (Adventure Seekers):** Young-Middle (25-45), Medium-High budget (3000-7000), Medium stay (5-15 days)
- **Cluster 4 (Spiritual Tourists):** All ages, Low-Medium budget (1000-4000), Variable stay (3-20 days)

**Dataset Specifications:**
- Sample size: 5000 records
- Save as: `data/tourism_pokhara.csv`

### 1.2 Data Preprocessing
**File:** `preprocess.py`

**Steps:**
1. Load the CSV dataset
2. Handle missing values (drop or impute)
3. Encode categorical variable (Activity Preference)
4. Scale features (StandardScaler for K-Means)
5. Save processed data and scalers

**Deliverables:**
- Clean dataset ready for clustering
- Preprocessing pipeline

---

## 🎯 Phase 2: K-Means Clustering (5 marks)

### 2.1 Finding Optimal K
**File:** `clustering.py`

**Methods:**
1. **Elbow Method:** Plot inertia vs K values
2. **Silhouette Analysis:** Measure cluster separation
3. **Domain Knowledge:** Use business understanding

### 2.2 K-Means Implementation
```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=5,        # Optimal K
    init='k-means++',    # Smart initialization
    n_init=10,           # Run 10 times
    max_iter=300,
    random_state=42
)
clusters = kmeans.fit_predict(X_scaled)
```

### 2.3 Cluster Analysis
**For each cluster, calculate:**
- Mean/Median of each feature
- Cluster size (number of tourists)
- Cluster characteristics (profile)

**Deliverables:**
- Optimal K determination
- Trained K-Means model
- Cluster profiles

---

## 🤖 Phase 3: MLJAR AutoML (5 marks)

### 3.1 What is MLJAR AutoML?

**MLJAR AutoML** is an automated machine learning library that:
- Tries multiple algorithms automatically
- Performs feature engineering
- Tunes hyperparameters
- Compares model performance
- Creates ensemble models

### 3.2 Implementation
**File:** `automl_analysis.py`

```python
from automl import AutoML

# Configure AutoML
automl = AutoML(
    total_time_limit=300,      # 5 minutes
    algorithms=["Xgboost", "Random Forest", "LightGBM", "CatBoost"],
    train_ensemble=True,
    explain_level=2,
    save_path="automl_results"
)

# Train (using cluster labels as target)
automl.fit(X_train, y_train)
```

### 3.3 What AutoML Will Do:
1. **Automatically train multiple models:**
   - XGBoost
   - Random Forest
   - LightGBM
   - CatBoost
   - Neural Networks
   - Ensemble models

2. **Compare performance:**
   - Accuracy
   - F1-Score
   - ROC-AUC
   - Confusion Matrix

3. **Generate reports:**
   - Model rankings
   - Feature importance
   - Learning curves

**Deliverables:**
- AutoML results folder
- Model comparison report
- Best model saved

---

## 🖥️ Phase 4: Streamlit Dashboard (5 marks)

### 4.1 Dashboard Structure
**File:** `app.py`

**Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  🏔️ Tourism Clustering Dashboard - Pokhara             │
├─────────────────────────────────────────────────────────┤
│  [Sidebar: Tourist Input Form]                          │
│  - Age: [_______] years                                 │
│  - Budget: [_______] NPR                                │
│  - Duration: [_______] days                             │
│  - Activity: [Dropdown]                                 │
│  [Find My Cluster Button]                               │
├─────────────────────────────────────────────────────────┤
│  [Main Area: Results & Visualizations]                  │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │  Cluster Result │  │  Interactive Charts          │  │
│  │  (colored box)  │  │  - 2D/3D Cluster Plot        │  │
│  │  + Profile      │  │  - Cluster Distribution      │  │
│  └─────────────────┘  │  - Feature Comparison        │  │
│                       │  - AutoML Results            │  │
│                       └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Dashboard Features

**1. Input Section (Sidebar):**
- Age slider (18-70)
- Budget slider (500-10000 NPR)
- Duration slider (1-30 days)
- Activity preference dropdown
- "Find My Cluster" button

**2. Prediction Display:**
- Assigned cluster number
- Cluster name (e.g., "Budget Backpacker")
- Cluster profile description
- Similar tourists percentage

**3. Visualizations:**

**Tab 1: Cluster Overview**
- 2D scatter plot (PCA-reduced)
- Color-coded clusters
- Interactive hover information

**Tab 2: Cluster Distribution**
- Pie chart of cluster sizes
- Bar chart of cluster characteristics

**Tab 3: Feature Comparison**
- Radar chart comparing clusters
- Box plots for each feature

**Tab 4: AutoML Results**
- Model comparison leaderboard
- Feature importance chart
- Best model details

**4. Tourism Strategies:**
For each cluster, suggest:
- Targeted marketing approaches
- Recommended packages
- Preferred accommodations
- Activity suggestions

**Deliverables:**
- Fully functional Streamlit app
- Cluster assignment capability
- Interactive visualizations

---

## 📊 Phase 5: Interpretation & Strategies (5 marks)

### 5.1 Cluster Interpretation

**For each cluster, document:**

| Cluster | Name | Characteristics | Size |
|---------|------|-----------------|------|
| 0 | Budget Backpackers | Young, low budget, short stay | ~25% |
| 1 | Cultural Explorers | Middle-aged, medium budget | ~20% |
| 2 | Luxury Travelers | Older, high budget, long stay | ~15% |
| 3 | Adventure Seekers | Active, medium-high budget | ~25% |
| 4 | Spiritual Tourists | All ages, pilgrimage focused | ~15% |

### 5.2 Tourism Strategies

**For Each Cluster:**

**Budget Backpackers:**
- Hostels and budget accommodations
- Group tour packages
- Street food tours
- Budget adventure activities

**Cultural Explorers:**
- Heritage site packages
- Local experience tours
- Museum discounts
- Cultural show tickets

**Luxury Travelers:**
- 5-star resort packages
- Private guided tours
- Helicopter tours
- Spa and wellness packages

**Adventure Seekers:**
- Trekking packages
- Paragliding discounts
- White water rafting
- Mountain biking tours

**Spiritual Tourists:**
- Temple circuit packages
- Meditation retreats
- Pilgrimage guides
- Peaceful accommodations

### 5.3 Business Recommendations

1. **Marketing:** Target each cluster with personalized ads
2. **Pricing:** Dynamic pricing based on cluster willingness to pay
3. **Packages:** Create cluster-specific tour packages
4. **Seasonality:** Adjust strategies based on tourist season

**Deliverables:**
- Cluster interpretation document
- Tourism strategy recommendations
- Business insights report

---

## 📁 Project File Structure

```
F:\ML project\
├── data/
│   └── tourism_pokhara.csv          # Synthetic dataset
├── documentations/
│   └── tourism_project_plan.md      # This document
├── automl_results/                  # AutoML output folder
├── kmeans_model.pkl                 # Trained K-Means model
├── scaler.pkl                       # Feature scaler
├── encoder.pkl                      # Categorical encoder
├── data_generator.py                # Dataset creation script
├── preprocess.py                    # Preprocessing functions
├── clustering.py                    # K-Means implementation
├── automl_analysis.py               # MLJAR AutoML script
├── app.py                           # Streamlit dashboard
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## 🛠️ Required Libraries

**requirements.txt:**
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
streamlit>=1.28.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0        # Interactive charts
joblib>=1.3.0
scikit-plot>=0.3.7    # Additional plotting
mljar-supervised>=1.0 # MLJAR AutoML
```

---

## ✅ Task Checklist

| Task | Status | File(s) | Marks |
|------|--------|---------|-------|
| Generate synthetic dataset | ⬜ | `data_generator.py` | 1 |
| Preprocess data (scaling, encoding) | ⬜ | `preprocess.py` | 2 |
| Apply K-Means Clustering | ⬜ | `clustering.py` | 2 |
| Run MLJAR AutoML | ⬜ | `automl_analysis.py` | 3 |
| Build Streamlit dashboard | ⬜ | `app.py` | 3 |
| Add visualizations (2D/3D plots, charts) | ⬜ | `app.py` | 2 |
| Interpret clusters and suggest strategies | ⬜ | `app.py`, docs | 2 |

---

## 🚀 Execution Order

1. **Step 1:** Run `data_generator.py` → creates dataset
2. **Step 2:** Run `clustering.py` → applies K-Means, saves model
3. **Step 3:** Run `automl_analysis.py` → runs AutoML (takes 5-10 min)
4. **Step 4:** Run `streamlit run app.py` → launches dashboard

---

## 📝 Evaluation Criteria Mapping

| Criteria | Implementation | Location |
|----------|----------------|----------|
| Load & preprocess | Pandas, StandardScaler, LabelEncoder | `clustering.py` |
| Apply K-Means Clustering | sklearn KMeans | `clustering.py` |
| Use MLJAR AutoML | AutoML class | `automl_analysis.py` |
| Dashboard functionality | Streamlit input + cluster assignment | `app.py` |
| Visualizations & interpretation | Charts, cluster profiles, strategies | `app.py` |

---

## 🔮 Key Differences from Project 1

| Aspect | Project 1 (Air Quality) | Project 2 (Tourism) |
|--------|------------------------|---------------------|
| **ML Type** | Supervised (Classification) | Unsupervised (Clustering) + Supervised (AutoML) |
| **Algorithm** | Decision Tree | K-Means + Multiple (AutoML) |
| **Output** | Category (Good/Moderate/Unhealthy) | Cluster Number (0-4) |
| **AutoML** | Not used | MLJAR for model comparison |
| **Focus** | Prediction | Segmentation + Interpretation |

---

## ❓ Queries / Decisions Needed

Before we start, please confirm:

1. **Dataset size:** 5000 records okay?
2. **Number of clusters:** 5 clusters reasonable, or want different?
3. **AutoML time limit:** 5 minutes okay for AutoML?
4. **Dashboard style:** Similar clean design as Project 1?

---

**Ready to begin? Let me know and I'll start with Phase 1!** 🚀
