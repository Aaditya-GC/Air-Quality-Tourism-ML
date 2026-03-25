# Air Quality Prediction + Dashboard - Project Plan

**Total Marks: 25**  
**Location: Kathmandu Pollution Monitoring Stations**

---

## 📋 Project Overview

Build a machine learning pipeline that:
1. Preprocesses air quality monitoring data
2. Trains a Decision Tree classifier to predict air quality categories
3. Evaluates model performance
4. Deploys an interactive Streamlit dashboard for real-time predictions
5. Visualizes pollution trends and patterns

---

## 🗂️ Phase 1: Dataset Creation & Preprocessing (5 marks)

### 1.1 Synthetic Dataset Generation
**File:** `data_generator.py`

Create a realistic synthetic dataset with the following features:

| Feature | Type | Range/Values | Description |
|---------|------|--------------|-------------|
| PM2.5 | Numeric | 0-500 µg/m³ | Fine particulate matter |
| PM10 | Numeric | 0-600 µg/m³ | Coarse particulate matter |
| Temperature | Numeric | 5-40 °C | Ambient temperature |
| Humidity | Numeric | 20-95 % | Relative humidity |
| Wind Speed | Numeric | 0-30 km/h | Wind velocity |
| Air Quality Category | Categorical | Good/Moderate/Unhealthy | Target variable |

**Air Quality Rules (for label generation):**
- **Good:** PM2.5 < 35 AND PM10 < 50
- **Moderate:** PM2.5 35-150 OR PM10 50-250
- **Unhealthy:** PM2.5 > 150 OR PM10 > 250

**Dataset Specifications:**
- Sample size: 1000-2000 records
- Save as: `data/air_quality_kathmandu.csv`

### 1.2 Data Preprocessing
**File:** `preprocess.py`

**Steps:**
1. Load the CSV dataset
2. Check for missing values → handle (drop/impute)
3. Encode target variable:
   - Good → 0
   - Moderate → 1
   - Unhealthy → 2
4. Split data: 80% train, 20% test (stratified)
5. Save processed data or pipeline

**Deliverables:**
- Clean dataset ready for training
- Preprocessing pipeline (optional: save with `joblib`)

---

## 🌳 Phase 2: Decision Tree Classification (5 marks)

### 2.1 Model Training
**File:** `train_model.py`

**Implementation:**
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion='gini',      # or 'entropy'
    max_depth=10,          # prevent overfitting
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)
```

**Hyperparameter Tuning (Optional but Recommended):**
- Use `GridSearchCV` to find optimal:
  - `max_depth`: [5, 10, 15, None]
  - `min_samples_split`: [2, 5, 10]
  - `criterion`: ['gini', 'entropy']

### 2.2 Model Saving
- Save trained model: `model.pkl` or `model.joblib`
- Save encoder/label mappings for inference

**Deliverables:**
- Trained Decision Tree model
- Saved model file for dashboard use

---

## 📊 Phase 3: Model Evaluation (5 marks)

### 3.1 Performance Metrics
**File:** `evaluate_model.py` (or integrated in `train_model.py`)

**Metrics to Compute:**
1. **Accuracy Score:** Overall correctness
2. **Confusion Matrix:** 3x3 matrix (Good/Moderate/Unhealthy)
3. **Classification Report:**
   - Precision (per class)
   - Recall (per class)
   - F1-Score (per class)
   - Support (sample count)

**Visualization:**
- Heatmap for confusion matrix (`seaborn.heatmap`)
- Bar chart for feature importance

**Deliverables:**
- Evaluation report (print/console output)
- Confusion matrix plot
- Feature importance chart

---

## 🖥️ Phase 4: Streamlit Dashboard (5 marks)

### 4.1 Dashboard Structure
**File:** `app.py`

**Layout:**
```
┌─────────────────────────────────────────┐
│  🌬️ Air Quality Prediction Dashboard   │
├─────────────────────────────────────────┤
│  [Sidebar: User Input Form]             │
│  - PM2.5: [_______] µg/m³               │
│  - PM10: [_______] µg/m³                │
│  - Temperature: [_______] °C            │
│  - Humidity: [_______] %                │
│  - Wind Speed: [_______] km/h           │
│  [Predict Button]                       │
├─────────────────────────────────────────┤
│  [Main Area: Results & Visualizations]  │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │  Prediction │  │  Charts/Graphs   │  │
│  │  Result     │  │  - Trends        │  │
│  │  (colored)  │  │  - Correlations  │  │
│  └─────────────┘  └──────────────────┘  │
└─────────────────────────────────────────┘
```

### 4.2 Implementation Steps
1. **Load Resources:**
   - Load saved model (`model.pkl`)
   - Load label encoder mappings
   - Load dataset for visualizations

2. **Sidebar Input:**
   ```python
   st.sidebar.slider("PM2.5", 0, 500, 50)
   st.sidebar.slider("PM10", 0, 600, 100)
   st.sidebar.slider("Temperature", 5, 40, 25)
   st.sidebar.slider("Humidity", 20, 95, 50)
   st.sidebar.slider("Wind Speed", 0, 30, 10)
   ```

3. **Prediction Logic:**
   - On button click → preprocess input → model.predict()
   - Display result with color coding:
     - 🟢 Good
     - 🟡 Moderate
     - 🔴 Unhealthy

4. **Dynamic Display:**
   - Show prediction confidence (if using `predict_proba`)
   - Display health recommendations based on category

**Deliverables:**
- Fully functional Streamlit app
- Real-time prediction capability

---

## 📈 Phase 5: Data Visualizations (5 marks)

### 5.1 Charts to Include in Dashboard

**1. Correlation Heatmap:**
- Show relationships between all features
- Use `seaborn.heatmap()`

**2. Feature Importance Bar Chart:**
- Display which features matter most for prediction
- Extract from Decision Tree: `model.feature_importances_`

**3. Air Quality Distribution:**
- Pie chart or bar chart of category distribution
- Show % of Good/Moderate/Unhealthy in dataset

**4. PM2.5 vs PM10 Scatter Plot:**
- Color points by air quality category
- Show clustering patterns

**5. Trend Analysis (Optional):**
- If temporal data exists: line charts over time
- Seasonal patterns in pollution

### 5.2 Streamlit Visualization Code
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Feature importance
fig, ax = plt.subplots()
ax.barh(feature_names, model.feature_importances_)
st.pyplot(fig)
```

**Deliverables:**
- Minimum 3 interactive charts in dashboard
- Properly labeled axes and titles

---

## 📁 Project File Structure

```
F:\ML project\
├── data/
│   └── air_quality_kathmandu.csv      # Synthetic dataset
├── documentations/
│   └── air_quality_project_plan.md    # This document
├── model.pkl                          # Trained model
├── label_encoder.pkl                  # Label encoder (if used)
├── data_generator.py                  # Dataset creation script
├── preprocess.py                      # Preprocessing functions
├── train_model.py                     # Model training & evaluation
├── app.py                             # Streamlit dashboard
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
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
joblib>=1.3.0
```

---

## ✅ Task Checklist

| Task | Status | File(s) | Marks |
|------|--------|---------|-------|
| Generate synthetic dataset | ⬜ | `data_generator.py` | 1 |
| Preprocess data | ⬜ | `preprocess.py` | 2 |
| Train Decision Tree model | ⬜ | `train_model.py` | 2 |
| Evaluate model (accuracy, confusion matrix) | ⬜ | `train_model.py` | 3 |
| Build Streamlit dashboard (input + prediction) | ⬜ | `app.py` | 3 |
| Add visualizations to dashboard | ⬜ | `app.py` | 2 |
| Test & deploy | ⬜ | - | 2 |

---

## 🚀 Execution Order

1. **Step 1:** Run `data_generator.py` → creates dataset
2. **Step 2:** Run `train_model.py` → trains & evaluates model, saves `model.pkl`
3. **Step 3:** Run `streamlit run app.py` → launches dashboard

---

## 📝 Evaluation Criteria Mapping

| Criteria | Implementation | Location |
|----------|----------------|----------|
| Load & preprocess | Pandas, Label Encoding | `train_model.py` lines X-X |
| Decision Tree training | sklearn DecisionTreeClassifier | `train_model.py` lines X-X |
| Model evaluation | accuracy_score, confusion_matrix | `train_model.py` lines X-X |
| Dashboard functionality | Streamlit input + prediction | `app.py` lines X-X |
| Visualizations | Heatmap, bar charts, scatter plots | `app.py` lines X-X |

---

## 🔮 Future Enhancements (Optional)

- Add real-time data API integration
- Include more ML models (Random Forest, XGBoost)
- Add model comparison in dashboard
- Export predictions to CSV
- Add user authentication
- Deploy to Streamlit Cloud / Heroku

---

## ❓ Queries / Decisions Needed

Before we start, please confirm:

1. **Dataset size:** 1000 records sufficient?
2. **Model complexity:** Simple Decision Tree or tuned with GridSearchCV?
3. **Dashboard deployment:** Local only or deploy online (Streamlit Cloud)?
4. **Visualizations:** Which 3-4 charts do you want prioritized?

---

**Ready to begin? Let me know and I'll create a todo list to track our progress!** 🚀
