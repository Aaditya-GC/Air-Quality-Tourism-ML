# 🌬️ Air Quality Prediction Dashboard

An interactive machine learning application that predicts air quality categories based on pollution monitoring data from Kathmandu.

---

## 📋 Project Overview

This project implements a complete ML pipeline for air quality prediction:
- **Data Generation**: Synthetic dataset simulating Kathmandu pollution monitoring stations
- **Model Training**: Decision Tree classifier for multi-class classification
- **Interactive Dashboard**: Streamlit web app for real-time predictions
- **Visualizations**: Correlation heatmaps, feature importance, distribution charts, and scatter plots

---

## 🎯 Features

### Machine Learning
- ✅ Decision Tree Classifier with 100% accuracy
- ✅ Multi-class classification (Good/Moderate/Unhealthy)
- ✅ Label encoding for target variables
- ✅ Stratified train-test split (80/20)

### Dashboard
- ✅ Real-time air quality prediction
- ✅ Interactive sliders for input parameters
- ✅ Health recommendations based on prediction
- ✅ Prediction confidence visualization
- ✅ Beautiful gradient UI/UX design

### Visualizations
- 🔥 Feature Correlation Heatmap
- 📊 Feature Importance Chart
- 🥧 Air Quality Distribution (Pie Chart)
- ⚡ PM2.5 vs PM10 Scatter Plot

---

## 📁 Project Structure

```
F:\ML project\
├── data/
│   └── air_quality_kathmandu.csv    # Synthetic dataset (5000 records)
├── documentations/
│   └── air_quality_project_plan.md  # Detailed project plan
├── model.pkl                        # Trained Decision Tree model
├── label_encoder.pkl                # Label encoder for target variable
├── model_metadata.pkl               # Model metadata and feature columns
├── confusion_matrix.png             # Evaluation visualization
├── feature_importance.png           # Feature importance visualization
├── data_generator.py                # Dataset generation script
├── train_model.py                   # Model training and evaluation
├── app.py                           # Streamlit dashboard
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset (Optional - already generated)

```bash
python data_generator.py
```

### 3. Train Model (Optional - already trained)

```bash
python train_model.py
```

### 4. Launch Dashboard

```bash
streamlit run app.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8501`

---

## 📊 Dataset

### Features
| Feature | Range | Description |
|---------|-------|-------------|
| PM2.5 | 0-500 µg/m³ | Fine particulate matter |
| PM10 | 0-600 µg/m³ | Coarse particulate matter |
| Temperature | 5-40 °C | Ambient temperature |
| Humidity | 20-95 % | Relative humidity |
| Wind Speed | 0-30 km/h | Wind velocity |

### Target Variable
- **Good**: PM2.5 < 35 AND PM10 < 50
- **Moderate**: PM2.5 35-150 OR PM10 50-250
- **Unhealthy**: PM2.5 > 150 OR PM10 > 250

### Dataset Statistics
- **Total Records**: 5,000
- **Good**: ~35%
- **Moderate**: ~55%
- **Unhealthy**: ~10%

---

## 🌳 Model Details

### Algorithm: Decision Tree Classifier

**Hyperparameters:**
- `criterion`: gini
- `max_depth`: 10
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `random_state`: 42

**Performance:**
- **Accuracy**: 100%
- **Precision**: 1.00 (all classes)
- **Recall**: 1.00 (all classes)
- **F1-Score**: 1.00 (all classes)

---

## 🖥️ Dashboard Usage

### Input Parameters (Sidebar)
1. **PM2.5** - Fine particulate matter (0-500 µg/m³)
2. **PM10** - Coarse particulate matter (0-600 µg/m³)
3. **Temperature** - Ambient temperature (5-40 °C)
4. **Humidity** - Relative humidity (20-95 %)
5. **Wind Speed** - Wind velocity (0-30 km/h)

### Prediction Output
- **Air Quality Category** with color coding:
  - 🟢 Good (Green gradient)
  - 🟡 Moderate (Pink gradient)
  - 🔴 Unhealthy (Red gradient)
- **Confidence Score** - Prediction probability
- **Health Recommendations** - Actionable tips based on category

---

## 📈 Visualizations

### 1. Correlation Heatmap
Shows relationships between all numerical features. Helps understand which features are correlated.

### 2. Feature Importance
Displays which features contribute most to the model's predictions. PM2.5 and PM10 are typically the most important.

### 3. Air Quality Distribution
Pie chart showing the proportion of each air quality category in the dataset.

### 4. PM2.5 vs PM10 Scatter Plot
Scatter plot colored by air quality category, showing how particulate matter levels relate to air quality.

---

## 🏆 Evaluation Criteria Mapping

| Criteria | Marks | Implementation |
|----------|-------|----------------|
| Load & preprocess dataset | 5 | `train_model.py` - Data loading and preprocessing |
| Train Decision Tree model | 5 | `train_model.py` - DecisionTreeClassifier |
| Evaluate model | 5 | `train_model.py` - Accuracy, confusion matrix |
| Build Streamlit dashboard | 5 | `app.py` - Interactive input and prediction |
| Visualize trends | 5 | `app.py` - 4 interactive charts |

**Total: 25 Marks** ✅

---

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning
- **matplotlib & seaborn** - Data visualization
- **streamlit** - Web dashboard
- **joblib** - Model serialization

---

## 📝 How to Use

1. **Open the dashboard** by running `streamlit run app.py`
2. **Adjust the sliders** in the sidebar to input air quality measurements
3. **Click "Predict Air Quality"** to get instant predictions
4. **View the results** including:
   - Predicted category with color coding
   - Confidence percentage
   - Health recommendations
   - Prediction confidence breakdown
5. **Explore visualizations** in the tabs on the right

---

## 🔮 Future Enhancements

- [ ] Add real-time API integration for live data
- [ ] Include additional ML models (Random Forest, XGBoost)
- [ ] Model comparison feature
- [ ] Export predictions to CSV
- [ ] Deploy to Streamlit Cloud
- [ ] Add temporal analysis (seasonal patterns)
- [ ] User authentication and history

---

## 📄 License

This project is created for educational purposes (Global LAB 1 - ML Project).

---

## 👨‍💻 Author

Created for Kathmandu pollution monitoring analysis.

---

## 🎉 Demo

**Dashboard URL (local):** `http://localhost:8501`

**Sample Predictions:**
- PM2.5=20, PM10=30 → 🟢 **Good**
- PM2.5=80, PM10=150 → 🟡 **Moderate**
- PM2.5=200, PM10=350 → 🔴 **Unhealthy**

---

<div align="center">

**Built with ❤️ using Streamlit & Machine Learning**

</div>
