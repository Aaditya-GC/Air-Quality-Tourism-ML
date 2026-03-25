# Global LAB 1 - Machine Learning Projects

**Name:** Aaditya GC  
**Roll No:** 2  
**Date:** March 2026

---

## What's Included

This submission contains two complete machine learning projects:

1. **Air Quality Prediction** - Predicts air quality using Decision Tree
2. **Tourism Clustering** - Segments tourists using K-Means and AutoML

---

## Project Structure

```
ML project/
├── photo.JPG                     (My photo)
├── GLOBAL_LAB1_README.md        (This file)
│
├── Project 1 - Air Quality/
│   ├── data/
│   │   └── air_quality_kathmandu.csv
│   ├── models/
│   │   ├── model.pkl
│   │   ├── label_encoder.pkl
│   │   └── model_metadata.pkl
│   ├── documentations/
│   │   └── Study/
│   │       └── Study-Guide-Air-Quality.md
│   ├── data_generator.py
│   ├── train_model.py
│   ├── app.py                    (Dashboard)
│   ├── requirements.txt
│   └── README.md
│
└── Project 2 - Tourism/
    ├── data/
    │   ├── tourism_pokhara.csv
    │   └── tourism_pokhara_clustered.csv
    ├── models/
    │   ├── kmeans_model.pkl
    │   ├── scaler.pkl
    │   ├── activity_encoder.pkl
    │   ├── model_metadata.pkl
    │   ├── automl_best_model.pkl
    │   ├── all_trained_models.pkl
    │   └── automl_results_summary.pkl
    ├── documentations/
    │   └── Study/
    │       └── Study-Guide-Tourism.md
    ├── tourism_data_generator.py
    ├── tourism_clustering.py
    ├── tourism_automl_analysis.py
    ├── tourism_app.py             (Dashboard)
    ├── tourism_requirements.txt
    └── README_TOURISM.md
```

---

## How to Run

### Install Python Requirements

**Project 1:**
```bash
cd "Project 1 - Air Quality"
pip install -r requirements.txt
```

**Project 2:**
```bash
cd "Project 2 - Tourism"
pip install -r tourism_requirements.txt
```

### Run the Dashboards

**Project 1 - Air Quality:**
```bash
cd "Project 1 - Air Quality"
streamlit run app.py
```
Then open: http://localhost:8501

**Project 2 - Tourism:**
```bash
cd "Project 2 - Tourism"
streamlit run tourism_app.py
```
Then open: http://localhost:8502

---

## Project Details

### Project 1: Air Quality Prediction

**What it does:** Predicts air quality (Good/Moderate/Unhealthy) from pollution data

**Algorithm:** Decision Tree Classifier  
**Accuracy:** 100%

**Features:**
- PM2.5, PM10, Temperature, Humidity, Wind Speed
- 5000 synthetic records
- Health recommendations for each prediction

**Visualizations:**
- Correlation heatmap
- Feature importance
- Air quality distribution
- PM2.5 vs PM10 scatter plot

---

### Project 2: Tourism Clustering

**What it does:** Segments tourists into 5 clusters and provides tourism strategies

**Algorithms:** 
- K-Means Clustering (5 clusters)
- AutoML (8 models compared)

**Results:**
- Silhouette Score: 0.338
- Best AutoML Model: Neural Network (99.6% accuracy)

**5 Tourist Clusters:**
1. Luxury Travelers (738 tourists)
2. Budget Backpackers (1,310 tourists)
3. Digital Nomads (52 tourists)
4. Adventure Seekers (1,268 tourists)
5. Cultural Explorers (1,632 tourists)

**Visualizations:**
- Cluster distribution
- Cluster characteristics
- PCA 2D projection
- AutoML model comparison

---

## Documentation

Each project has a complete study guide in the `documentations/Study/` folder:

- **Project 1:** `Study-Guide-Air-Quality.md`
- **Project 2:** `Study-Guide-Tourism.md`

These guides include:
- Complete code explanations
- ML concepts explained
- Results and evaluation
- Report writing guide

---

## Notes for Teacher

- Both dashboards have my photo and details in the sidebar
- All code is my own work
- Documentation is comprehensive but you can refer to the study guides for details
- Models are pre-trained and ready to use

---

**Thank you for reviewing my submission!**
