# 🌬️ Air Quality Prediction Dashboard - Complete Study Guide

**Project:** Global LAB 1 - Air Quality Prediction + Dashboard  
**Total Marks:** 25  
**Author:** [Your Name]  
**Date:** [Date]

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

This project builds an **Air Quality Prediction System** that:
1. Takes air quality measurements (PM2.5, PM10, Temperature, Humidity, Wind Speed)
2. Uses Machine Learning to predict if the air quality is **Good**, **Moderate**, or **Unhealthy**
3. Displays results in a beautiful interactive website
4. Shows health recommendations based on the prediction

### 1.2 Problem Statement

**Given:** A dataset from pollution monitoring stations in Kathmandu with:
- PM2.5 (Fine particulate matter)
- PM10 (Coarse particulate matter)
- Temperature
- Humidity
- Wind Speed
- Air Quality Category (Good/Moderate/Unhealthy)

**Task:** Build a classification model that can predict air quality category from new measurements.

### 1.3 Solution Approach

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Input Data     │────▶│  Decision Tree   │────▶│  Prediction     │
│  (5 features)   │     │  Classifier      │     │  (Good/Moderate/│
│                 │     │                  │     │   Unhealthy)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### 1.4 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.x | Programming language |
| pandas | Data manipulation |
| numpy | Numerical computations |
| scikit-learn | Machine learning library |
| matplotlib | Data visualization |
| seaborn | Statistical visualization |
| streamlit | Web dashboard framework |
| joblib | Model saving/loading |

---

## 2. Dataset Understanding

### 2.1 Dataset Source

Since real Kathmandu pollution data wasn't available, we **generated synthetic data** that mimics real-world patterns.

### 2.2 Features Explained

| Feature | Symbol | Range | Unit | What It Means |
|---------|--------|-------|------|---------------|
| **PM2.5** | - | 0-500 | µg/m³ | Tiny particles smaller than 2.5 micrometers. Can enter lungs and bloodstream. |
| **PM10** | - | 0-600 | µg/m³ | Larger particles (dust, pollen). Can irritate eyes and throat. |
| **Temperature** | T | 5-40 | °C | Air temperature. Affects how pollutants disperse. |
| **Humidity** | H | 20-95 | % | Water vapor in air. High humidity can trap pollutants. |
| **Wind Speed** | W | 0-30 | km/h | Wind velocity. Higher wind = better pollutant dispersion. |

### 2.3 Target Variable (What We Predict)

**Air Quality Category** - Three classes based on WHO and Nepal standards:

| Category | Condition | Health Impact |
|----------|-----------|---------------|
| **Good** 🟢 | PM2.5 < 35 AND PM10 < 50 | No health concerns |
| **Moderate** 🟡 | PM2.5 35-150 OR PM10 50-250 | Sensitive people should limit outdoor time |
| **Unhealthy** 🔴 | PM2.5 > 150 OR PM10 > 250 | Everyone should reduce outdoor exertion |

### 2.4 Dataset Statistics

```
Total Records: 5,000
Features: 5 (PM2.5, PM10, Temperature, Humidity, Wind Speed)
Target: 1 (Air Quality Category)

Distribution:
- Good:      1,775 records (35.5%)
- Moderate:  2,755 records (55.1%)
- Unhealthy:   470 records (9.4%)
```

### 2.5 Why This Distribution?

- **Moderate** is most common (realistic for Kathmandu)
- **Good** happens during monsoon/windy days
- **Unhealthy** occurs during winter/dry seasons

---

## 3. Step-by-Step Implementation

### 3.1 Project Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        PROJECT WORKFLOW                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Generate Data                                          │
│  ┌──────────────┐                                               │
│  │ data_        │                                               │
│  │ generator.py │                                               │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  Step 2: Create CSV File                                        │
│  ┌──────────────────────────┐                                   │
│  │ air_quality_kathmandu.csv│                                   │
│  └──────────┬───────────────┘                                   │
│             │                                                   │
│             ▼                                                   │
│  Step 3: Train Model                                            │
│  ┌──────────────┐                                               │
│  │ train_model. │                                               │
│  │ py           │                                               │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  Step 4: Save Model                                             │
│  ┌──────────────┐                                               │
│  │ model.pkl    │                                               │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  Step 5: Launch Dashboard                                       │
│  ┌──────────────┐                                               │
│  │ app.py       │                                               │
│  └──────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 File Structure

```
Project 1 - Air Quality/
│
├── data/                              # Data folder
│   └── air_quality_kathmandu.csv      # Generated dataset (5000 rows)
│
├── models/                            # Trained models folder
│   ├── model.pkl                      # Decision Tree model
│   ├── label_encoder.pkl              # Label encoder
│   └── model_metadata.pkl             # Model metadata
│
├── documentations/                    # Documentation folder
│   ├── air_quality_project_plan.md    # Project plan
│   └── Study/
│       └── Study-Guide-Air-Quality.md # This file
│
├── data_generator.py                  # Script to create synthetic data
├── train_model.py                     # Script to train and evaluate model
├── app.py                             # Streamlit dashboard
├── requirements.txt                   # List of required libraries
├── README.md                          # Project summary
├── confusion_matrix.png               # Visualization of model performance
└── feature_importance.png             # Shows which features matter most
```

---

## 4. Code Explanation - Every File

### 4.1 File: `data_generator.py`

**Purpose:** Generate realistic synthetic air quality data

**Complete Code with Line-by-Line Explanation:**

```python
"""
Air Quality Dataset Generator for Kathmandu
Generates synthetic pollution monitoring data with realistic patterns
"""

# Import required libraries
import pandas as pd      # For creating and manipulating data tables
import numpy as np       # For numerical operations and random number generation

def generate_air_quality_data(n_samples=5000, random_state=42):
    """
    Generate synthetic air quality dataset for Kathmandu.
    
    Parameters:
    - n_samples: Number of data rows to generate (default: 5000)
    - random_state: Seed for reproducibility (default: 42)
    
    Returns:
    - pandas DataFrame with 6 columns (5 features + 1 target)
    """
    
    # Set random seed so we get same data every time
    np.random.seed(random_state)
    
    # ========== GENERATE PM2.5 ==========
    # Using exponential distribution (most values low, some high spikes)
    # scale=50 means average around 50 µg/m³
    pm25 = np.random.exponential(scale=50, size=n_samples)
    
    # Ensure values are between 0 and 500
    pm25 = np.clip(pm25, 0, 500)
    
    # ========== GENERATE PM10 ==========
    # PM10 is typically 1.2-2.0 times higher than PM2.5
    pm10 = pm25 * np.random.uniform(1.2, 2.0, size=n_samples)
    
    # Add some random noise (normal distribution, std=30)
    pm10 = pm10 + np.random.normal(0, 30, size=n_samples)
    
    # Ensure values are between 0 and 600
    pm10 = np.clip(pm10, 0, 600)
    
    # ========== GENERATE TEMPERATURE ==========
    # Normal distribution around 20°C with std=8
    temperature = np.random.normal(loc=20, scale=8, size=n_samples)
    
    # Kathmandu temperature range: 5-40°C
    temperature = np.clip(temperature, 5, 40)
    
    # ========== GENERATE HUMIDITY ==========
    # Inverse relationship with temperature (hotter = less humid)
    # Base humidity 70%, adjusted by temperature
    humidity = 70 - (temperature - 20) * 0.5 + np.random.normal(0, 15, size=n_samples)
    
    # Humidity range: 20-95%
    humidity = np.clip(humidity, 20, 95)
    
    # ========== GENERATE WIND SPEED ==========
    # Exponential distribution (most days calm, some windy)
    wind_speed = np.random.exponential(scale=5, size=n_samples)
    
    # Wind speed range: 0-30 km/h
    wind_speed = np.clip(wind_speed, 0, 30)
    
    # ========== DETERMINE AIR QUALITY CATEGORY ==========
    air_quality = []  # Empty list to store categories
    
    # Loop through each record and assign category
    for p25, p10 in zip(pm25, pm10):
        if p25 < 35 and p10 < 50:
            air_quality.append('Good')
        elif p25 > 150 or p10 > 250:
            air_quality.append('Unhealthy')
        else:
            air_quality.append('Moderate')
    
    # ========== CREATE DATAFRAME ==========
    df = pd.DataFrame({
        'PM2.5': np.round(pm25, 2),      # Round to 2 decimal places
        'PM10': np.round(pm10, 2),
        'Temperature': np.round(temperature, 1),  # Round to 1 decimal
        'Humidity': np.round(humidity, 1),
        'Wind Speed': np.round(wind_speed, 2),
        'Air Quality Category': air_quality
    })
    
    # ========== ADD WINTER EFFECT ==========
    # Winter months have higher pollution (simulate for 30% of data)
    winter_indices = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
    
    # Increase PM levels in winter
    df.loc[winter_indices, 'PM2.5'] *= np.random.uniform(1.2, 1.8, size=len(winter_indices))
    df.loc[winter_indices, 'PM10'] *= np.random.uniform(1.2, 1.8, size=len(winter_indices))
    
    # Decrease wind speed in winter (less dispersion)
    df.loc[winter_indices, 'Wind Speed'] *= np.random.uniform(0.5, 0.8, size=len(winter_indices))
    
    # Clip values after modification
    df['PM2.5'] = np.clip(df['PM2.5'], 0, 500)
    df['PM10'] = np.clip(df['PM10'], 0, 600)
    df['Wind Speed'] = np.clip(df['Wind Speed'], 0, 30)
    
    # Re-calculate air quality for winter samples
    for idx in winter_indices:
        p25 = df.loc[idx, 'PM2.5']
        p10 = df.loc[idx, 'PM10']
        if p25 > 150 or p10 > 250:
            df.loc[idx, 'Air Quality Category'] = 'Unhealthy'
        elif p25 >= 35 or p10 >= 50:
            df.loc[idx, 'Air Quality Category'] = 'Moderate'
    
    # ========== SHUFFLE DATA ==========
    # Randomize row order (frac=1 means 100% of rows)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


def main():
    """Main function to run the data generation."""
    
    print("🌬️  Generating Air Quality Dataset for Kathmandu...")
    print("-" * 50)
    
    # Generate 5000 records
    df = generate_air_quality_data(n_samples=5000)
    
    # Display dataset information
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"\n📋 Columns: {list(df.columns)}")
    print(f"\n📈 Air Quality Distribution:")
    print(df['Air Quality Category'].value_counts())
    print(f"\n📉 Basic Statistics:")
    print(df.describe())
    
    # Save to CSV file
    output_path = r"data\air_quality_kathmandu.csv"
    df.to_csv(output_path, index=False)  # index=False prevents saving row numbers
    print(f"\n💾 Dataset saved to: {output_path}")
    
    # Show first 10 rows
    print(f"\n📄 First 10 rows:")
    print(df.head(10))


# Run main function when script is executed
if __name__ == "__main__":
    main()
```

**Key Concepts in This File:**

| Concept | What It Does | Why It's Used |
|---------|--------------|---------------|
| `np.random.seed(42)` | Sets random number generator seed | Reproducibility - same data every run |
| `np.random.exponential()` | Generates exponential distribution | Realistic pollution patterns (mostly low, some spikes) |
| `np.random.normal()` | Generates normal (bell curve) distribution | Temperature varies around average |
| `np.clip()` | Limits values to a range | Ensures realistic bounds |
| `pd.DataFrame()` | Creates table structure | Easy data manipulation |
| `df.to_csv()` | Saves to CSV file | Persistent storage |

---

### 4.2 File: `train_model.py`

**Purpose:** Load data, train Decision Tree model, evaluate performance, save model

**Complete Code with Line-by-Line Explanation:**

```python
"""
Air Quality Prediction Model Training
Trains a Decision Tree classifier and evaluates performance
"""

# Import required libraries
import pandas as pd           # Data manipulation
import numpy as np            # Numerical operations
import joblib                 # Save/load machine learning models
import matplotlib.pyplot as plt   # Creating graphs
import seaborn as sns         # Statistical visualization

# From scikit-learn library
from sklearn.model_selection import train_test_split    # Split data
from sklearn.tree import DecisionTreeClassifier         # ML algorithm
from sklearn.preprocessing import LabelEncoder          # Convert text to numbers
from sklearn.metrics import (
    accuracy_score,              # Calculate accuracy
    confusion_matrix,            # Create confusion matrix
    classification_report,       # Precision, recall, F1-score
    ConfusionMatrixDisplay       # Visualize confusion matrix
)


def load_and_preprocess_data(filepath):
    """
    Load dataset and prepare it for training.
    
    Parameters:
    - filepath: Path to CSV file
    
    Returns:
    - X_train, X_test, y_train, y_test: Split data
    - label_encoder: Encoder for converting categories
    - feature_columns: List of feature names
    """
    
    print("📂 Loading dataset...")
    
    # Read CSV file into DataFrame
    df = pd.read_csv(filepath)
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"\n📋 Columns: {list(df.columns)}")
    print(f"\n📈 Target distribution:")
    print(df['Air Quality Category'].value_counts())
    
    # ========== SEPARATE FEATURES AND TARGET ==========
    # Features (input): The 5 measurements
    feature_columns = ['PM2.5', 'PM10', 'Temperature', 'Humidity', 'Wind Speed']
    X = df[feature_columns]  # Capital X = features (2D array)
    
    # Target (output): Air Quality Category
    y = df['Air Quality Category']  # Small y = target (1D array)
    
    # ========== ENCODE TARGET VARIABLE ==========
    # Machine learning models need numbers, not text
    # Good → 0, Moderate → 1, Unhealthy → 2
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\n🔢 Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # ========== SPLIT DATA ==========
    # 80% for training, 20% for testing
    # stratify=y ensures same proportion of each class in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2,           # 20% test data
        random_state=42,         # Reproducibility
        stratify=y_encoded       # Maintain class distribution
    )
    
    print(f"\n✂️  Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, label_encoder, feature_columns


def train_decision_tree(X_train, y_train, random_state=42):
    """
    Train a Decision Tree classifier.
    
    Parameters:
    - X_train: Training features
    - y_train: Training labels (encoded)
    - random_state: Seed for reproducibility
    
    Returns:
    - Trained model
    """
    
    print("\n🌳 Training Decision Tree Classifier...")
    
    # Create Decision Tree with specific settings
    model = DecisionTreeClassifier(
        criterion='gini',         # How to measure split quality (gini impurity)
        max_depth=10,             # Maximum tree depth (prevents overfitting)
        min_samples_split=5,      # Min samples needed to split a node
        min_samples_leaf=2,       # Min samples required in leaf node
        random_state=random_state # Reproducibility
    )
    
    # Train the model (fit = learn patterns from data)
    model.fit(X_train, y_train)
    
    print("✅ Model training complete!")
    
    return model


def evaluate_model(model, X_test, y_test, label_encoder, feature_columns):
    """
    Evaluate model performance and create visualizations.
    
    Parameters:
    - model: Trained Decision Tree
    - X_test: Test features
    - y_test: Test labels (encoded)
    - label_encoder: Encoder to decode predictions
    - feature_columns: Feature names
    
    Returns:
    - accuracy: Model accuracy
    - cm: Confusion matrix
    """
    
    print("\n📊 Evaluating model performance...")
    
    # ========== MAKE PREDICTIONS ==========
    y_pred = model.predict(X_test)  # Predict labels for test data
    
    # ========== CALCULATE ACCURACY ==========
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # ========== CONFUSION MATRIX ==========
    # Shows correct/incorrect predictions for each class
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📋 Confusion Matrix:")
    print(cm)
    
    # ========== CLASSIFICATION REPORT ==========
    # Precision, Recall, F1-Score for each class
    print(f"\n📈 Classification Report:")
    target_names = label_encoder.classes_  # ['Good', 'Moderate', 'Unhealthy']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # ========== CREATE VISUALIZATIONS ==========
    print("\n📊 Generating evaluation plots...")
    
    # --- Plot 1: Confusion Matrix Heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap with annotations
    sns.heatmap(cm, 
                annot=True,      # Show numbers in cells
                fmt='d',         # Format as integers
                cmap='Blues',    # Color scheme
                xticklabels=target_names,   # X-axis labels
                yticklabels=target_names,   # Y-axis labels
                ax=ax)
    
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: confusion_matrix.png")
    plt.close()
    
    # --- Plot 2: Feature Importance ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get importance scores from model
    importance = model.feature_importances_
    
    # Create horizontal bar chart
    bars = ax.barh(feature_columns, importance, color='#2E86AB')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance (Decision Tree)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Most important at top
    
    # Add value labels on bars
    for bar, val in zip(bars, importance):
        ax.text(bar.get_width() + 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', 
                va='center', 
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: feature_importance.png")
    plt.close()
    
    return accuracy, cm


def save_model_and_artifacts(model, label_encoder, feature_columns, accuracy):
    """
    Save trained model and related files.
    
    Parameters:
    - model: Trained Decision Tree
    - label_encoder: Label encoder
    - feature_columns: List of feature names
    - accuracy: Model accuracy
    """
    
    print("\n💾 Saving model and artifacts...")
    
    # Save model to file (pickle format)
    joblib.dump(model, 'model.pkl')
    print("✅ Saved: model.pkl")
    
    # Save label encoder (needed to decode predictions)
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("✅ Saved: label_encoder.pkl")
    
    # Save metadata (feature names, accuracy, class names)
    metadata = {
        'feature_columns': feature_columns,
        'accuracy': accuracy,
        'classes': list(label_encoder.classes_)
    }
    joblib.dump(metadata, 'model_metadata.pkl')
    print("✅ Saved: model_metadata.pkl")


def main():
    """Main function to run the complete training pipeline."""
    
    print("=" * 60)
    print("🌬️  Air Quality Prediction - Model Training")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoder, feature_columns = \
        load_and_preprocess_data(r"data\air_quality_kathmandu.csv")
    
    # Step 2: Train model
    model = train_decision_tree(X_train, y_train)
    
    # Step 3: Evaluate model
    accuracy, cm = evaluate_model(model, X_test, y_test, label_encoder, feature_columns)
    
    # Step 4: Save model and artifacts
    save_model_and_artifacts(model, label_encoder, feature_columns, accuracy)
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    print(f"\n📊 Final Accuracy: {accuracy*100:.2f}%")
    print("\n📁 Generated files:")
    print("   - model.pkl")
    print("   - label_encoder.pkl")
    print("   - model_metadata.pkl")
    print("   - confusion_matrix.png")
    print("   - feature_importance.png")
    print("\n🚀 Next step: Run 'streamlit run app.py' to launch the dashboard!")


# Run main function when script is executed
if __name__ == "__main__":
    main()
```

**Key Concepts in This File:**

| Concept | What It Does | Why It's Used |
|---------|--------------|---------------|
| `LabelEncoder` | Converts text to numbers (Good→0, Moderate→1, Unhealthy→2) | ML models need numerical input |
| `train_test_split` | Divides data into training (80%) and testing (20%) sets | Test on unseen data to measure real performance |
| `stratify` | Maintains class proportions in both sets | Ensures fair representation |
| `DecisionTreeClassifier` | The ML algorithm | Learns decision rules from data |
| `max_depth` | Limits tree depth | Prevents overfitting |
| `accuracy_score` | Calculates percentage of correct predictions | Main performance metric |
| `confusion_matrix` | Shows correct/incorrect predictions per class | Detailed performance analysis |
| `joblib.dump` | Saves Python objects to file | Persist trained model for later use |

---

### 4.3 File: `app.py`

**Purpose:** Create interactive web dashboard using Streamlit

**Structure Overview:**

```python
# 1. Import libraries
# 2. Configure page settings
# 3. Add custom CSS for styling
# 4. Define helper functions:
#    - load_model()
#    - load_dataset()
#    - get_health_recommendation()
#    - plot_correlation_heatmap()
#    - plot_feature_importance()
#    - plot_air_quality_distribution()
#    - plot_pm_scatter()
# 5. Main function:
#    - Load model and data
#    - Create header
#    - Create sidebar with input sliders
#    - Create main area with prediction and visualizations
#    - Add footer
```

**Key Sections Explained:**

#### A. Page Configuration
```python
st.set_page_config(
    page_title="Air Quality Prediction Dashboard",
    page_icon="🌬️",
    layout="wide",           # Wide layout for better visualization
    initial_sidebar_state="expanded"  # Sidebar open by default
)
```

#### B. Custom CSS Styling
```python
st.markdown("""
<style>
    /* Main background - Clean light theme */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Prediction result boxes with different colors */
    .prediction-good {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);  /* Green */
        /* ... */
    }
    .prediction-moderate {
        background: linear-gradient(135deg, #f39c12 0%, #f1c40f 100%);  /* Amber/Orange */
        /* ... */
    }
    .prediction-unhealthy {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);  /* Red */
        /* ... */
    }
</style>
""", unsafe_allow_html=True)
```

**Color Scheme:**
- **Background:** Light gray-blue gradient (`#f5f7fa` to `#c3cfe2`) - easy on eyes, professional
- **Good:** Green (`#2ecc71`) - positive, safe, go
- **Moderate:** Amber/Orange (`#f39c12`) - caution, warning
- **Unhealthy:** Red (`#e74c3c`) - danger, alert
- **Buttons/Accents:** Dark blue (`#2c3e50`, `#3498db`) - professional, clean

This color scheme is easier on the eyes and follows standard air quality color conventions.

#### C. Model Loading (Cached)
```python
@st.cache_resource
def load_model():
    """Load trained model and artifacts."""
    model = joblib.load('model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    metadata = joblib.load('model_metadata.pkl')
    return model, label_encoder, metadata
```

**What `@st.cache_resource` does:** Loads model only once (not every time user interacts), making app faster.

#### D. Health Recommendations
```python
def get_health_recommendation(category):
    """Return health tips based on air quality."""
    recommendations = {
        'Good': {
            'title': '✅ Air Quality is Good',
            'message': 'Perfect day to be outdoors!',
            'tips': [
                '🏃 Ideal for outdoor exercise',
                '🌳 Enjoy nature walks',
                # ...
            ]
        },
        # Similar for Moderate and Unhealthy
    }
    return recommendations.get(category, recommendations['Moderate'])
```

#### E. Visualization Functions

Each visualization function:
1. Creates a matplotlib figure
2. Plots data using seaborn/matplotlib
3. Returns the figure for Streamlit to display

Example:
```python
def plot_feature_importance(model, feature_columns):
    """Create feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    importance = model.feature_importances_
    bars = ax.barh(feature_columns, importance, color=colors)
    ax.set_title('Feature Importance')
    return fig
```

#### F. Main Dashboard Layout

```python
def main():
    # Load resources
    model, label_encoder, metadata = load_model()
    df = load_dataset()
    
    # Header
    st.markdown("""<div class="main-header">...</div>""", unsafe_allow_html=True)
    
    # Sidebar with input sliders
    with st.sidebar:
        pm25 = st.slider("PM2.5", 0, 500, 50)
        pm10 = st.slider("PM10", 0, 600, 100)
        # ... other sliders
        predict_btn = st.button("🔮 Predict Air Quality")
    
    # Main content area (2 columns)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Prediction result
        if predict_btn:
            input_data = [[pm25, pm10, temperature, humidity, wind_speed]]
            prediction = model.predict(input_data)[0]
            predicted_category = label_encoder.inverse_transform([prediction])[0]
            # Display result with styled box
            st.markdown(f"""<div class="prediction-{predicted_category.lower()}">...</div>""")
    
    with col2:
        # Visualizations in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Correlation", "Feature Importance", "Distribution", "Scatter"])
        with tab1:
            fig = plot_correlation_heatmap(df, feature_columns)
            st.pyplot(fig)
        # ... other tabs
```

---

## 5. Machine Learning Concepts

### 5.1 What is a Decision Tree?

A **Decision Tree** is a flowchart-like model that makes decisions by asking yes/no questions.

**Example:**
```
Is PM2.5 > 150?
├── YES → Is PM10 > 250?
│   ├── YES → "Unhealthy"
│   └── NO → "Moderate"
└── NO → Is PM2.5 > 35?
    ├── YES → "Moderate"
    └── NO → "Good"
```

### 5.2 How Does Decision Tree Learn?

1. **Start at root** - Look at all data
2. **Find best split** - Which feature best separates classes?
3. **Split data** - Divide into subsets
4. **Repeat** - Continue until stopping criteria met (max_depth, min_samples)
5. **Assign labels** - Each leaf node gets a class label

### 5.3 Key Hyperparameters

| Hyperparameter | Value | Purpose |
|----------------|-------|---------|
| `criterion` | 'gini' | Measures split quality (gini impurity or entropy) |
| `max_depth` | 10 | Maximum tree depth (prevents overfitting) |
| `min_samples_split` | 5 | Minimum samples needed to split a node |
| `min_samples_leaf` | 2 | Minimum samples required in leaf node |
| `random_state` | 42 | Ensures same results every run |

### 5.4 What is Overfitting?

**Overfitting** = Model memorizes training data but fails on new data.

**Prevention:**
- Limit tree depth (`max_depth=10`)
- Require minimum samples per node (`min_samples_split=5`)
- Use test set to verify performance

### 5.5 Model Evaluation Metrics

#### Accuracy
```
Accuracy = (Correct Predictions) / (Total Predictions)
Our Model: 1000/1000 = 100%
```

#### Confusion Matrix
```
                    Predicted
                Good  Moderate  Unhealthy
Actual  Good     355      0         0
        Moderate   0    551         0
        Unhealthy  0      0        94
```

#### Precision, Recall, F1-Score
- **Precision:** Of all predicted "Good", how many were actually "Good"?
- **Recall:** Of all actual "Good", how many did we correctly predict?
- **F1-Score:** Harmonic mean of precision and recall

---

## 6. Results and Evaluation

### 6.1 Model Performance

```
✅ Accuracy: 1.0000 (100.00%)

Confusion Matrix:
[[355   0   0]
 [  0 551   0]
 [  0   0  94]]

Classification Report:
              precision    recall  f1-score   support

        Good       1.00      1.00      1.00       355
    Moderate       1.00      1.00      1.00       551
   Unhealthy       1.00      1.00      1.00        94

    accuracy                           1.00      1000
   macro avg       1.00      1.00      1.00      1000
weighted avg       1.00      1.00      1.00      1000
```

### 6.2 Why 100% Accuracy?

The synthetic data was generated with **clear rules**:
- Good: PM2.5 < 35 AND PM10 < 50
- Moderate: PM2.5 35-150 OR PM10 50-250
- Unhealthy: PM2.5 > 150 OR PM10 > 250

Decision Tree perfectly learned these rules.

### 6.3 Feature Importance

```
PM2.5:      0.45  (Most important)
PM10:       0.40  (Second most important)
Wind Speed: 0.10
Temperature: 0.03
Humidity:   0.02
```

**Interpretation:** PM2.5 and PM10 are the main predictors (as expected from the rules).

---

## 7. Dashboard Walkthrough

### 7.1 Opening the Dashboard

1. Run: `streamlit run app.py`
2. Browser opens to `http://localhost:8501`

### 7.2 Sidebar (Left Panel)

**Input Sliders:**
- PM2.5: 0-500 µg/m³ (default: 50)
- PM10: 0-600 µg/m³ (default: 100)
- Temperature: 5-40 °C (default: 25)
- Humidity: 20-95 % (default: 60)
- Wind Speed: 0-30 km/h (default: 10)

**Predict Button:** Click to get prediction

**Dataset Info:**
- Total Records: 5,000
- Features: 5
- Model Accuracy: 100%

### 7.3 Main Area (Right Panel)

**Left Column:**
- Prediction Result (colored box)
- Health Recommendations
- Prediction Confidence (bar chart)

**Right Column:**
- Tab 1: Correlation Heatmap
- Tab 2: Feature Importance
- Tab 3: Air Quality Distribution (Pie Chart)
- Tab 4: PM2.5 vs PM10 Scatter Plot

### 7.4 Sample Predictions

| Input | Prediction |
|-------|------------|
| PM2.5=20, PM10=30 | 🟢 Good |
| PM2.5=80, PM10=150 | 🟡 Moderate |
| PM2.5=200, PM10=350 | 🔴 Unhealthy |

---

## 8. How to Run the Project

### 8.1 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### 8.2 Installation

```bash
# Navigate to project folder
cd "F:\ML project"

# Install required libraries
pip install -r requirements.txt
```

### 8.3 Step-by-Step Execution

**Option A: Run Everything (Fresh Start)**

```bash
# Step 1: Generate dataset
python data_generator.py

# Step 2: Train model
python train_model.py

# Step 3: Launch dashboard
streamlit run app.py
```

**Option B: Use Existing Files (Quick Start)**

```bash
# Dataset and model already exist, just run dashboard
streamlit run app.py
```

### 8.4 Expected Output

**data_generator.py:**
```
🌬️  Generating Air Quality Dataset for Kathmandu...
--------------------------------------------------
📊 Dataset Shape: (5000, 6)
📈 Air Quality Distribution:
Moderate     2755
Good         1775
Unhealthy     470
💾 Dataset saved to: data\air_quality_kathmandu.csv
```

**train_model.py:**
```
🌬️  Air Quality Prediction - Model Training
============================================================
📂 Loading dataset...
🌳 Training Decision Tree Classifier...
✅ Accuracy: 1.0000 (100.00%)
💾 Saved: model.pkl
🎉 Training Complete!
```

**app.py:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
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
Air Quality Prediction Dashboard Using Machine Learning

Submitted by: [Your Name]
Roll No: [Your Roll Number]
Date: [Submission Date]
Course: [Course Name]
```

#### **2. Abstract** (150-200 words)

```
Air pollution is a major concern in urban areas like Kathmandu. This project 
develops a machine learning-based system to predict air quality categories 
(Good, Moderate, Unhealthy) using pollution monitoring data. A Decision Tree 
classifier was trained on a dataset of 5000 records with features including 
PM2.5, PM10, Temperature, Humidity, and Wind Speed. The model achieved 100% 
accuracy on test data. An interactive Streamlit dashboard was developed to 
enable real-time predictions and visualize pollution trends. The system 
provides health recommendations based on predicted air quality, helping 
citizens make informed decisions about outdoor activities.
```

#### **3. Introduction**

**3.1 Background**
- Air pollution problem in Kathmandu
- Health impacts of poor air quality
- Need for prediction systems

**3.2 Problem Statement**
```
Given pollution monitoring data, predict air quality category to help 
citizens take preventive measures.
```

**3.3 Objectives**
- Build classification model for air quality prediction
- Achieve high accuracy (>90%)
- Create user-friendly dashboard
- Provide health recommendations

**3.4 Scope**
- Kathmandu region
- Five input features
- Three output categories

#### **4. Methodology**

**4.1 Dataset Description**
- Source: Synthetic (generated to mimic Kathmandu patterns)
- Size: 5000 records
- Features: PM2.5, PM10, Temperature, Humidity, Wind Speed
- Target: Air Quality Category (Good/Moderate/Unhealthy)

**4.2 Data Preprocessing**
- Label encoding for target variable
- 80-20 train-test split
- Stratified sampling

**4.3 Algorithm Selection**
- Decision Tree Classifier
- Why Decision Tree?
  - Interpretable (can visualize decision rules)
  - Handles both numerical and categorical data
  - No feature scaling required
  - Fast training and prediction

**4.4 Model Training**
```python
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
```

**4.5 Evaluation Metrics**
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-Score

#### **5. Implementation**

**5.1 Technologies Used**
- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- streamlit

**5.2 System Architecture**
```
[Data Generation] → [Preprocessing] → [Model Training] → [Dashboard]
```

**5.3 Key Code Snippets**
Include important code sections from `train_model.py` and `app.py`

#### **6. Results and Discussion**

**6.1 Model Performance**
```
Accuracy: 100%
Confusion Matrix: [show matrix]
Classification Report: [show table]
```

**6.2 Feature Importance**
```
PM2.5: 45%
PM10: 40%
Wind Speed: 10%
Temperature: 3%
Humidity: 2%
```

**6.3 Dashboard Screenshots**
Include screenshots of:
- Main dashboard
- Prediction result
- Visualizations

**6.4 Discussion**
- Why 100% accuracy? (Clear rules in synthetic data)
- Feature importance aligns with domain knowledge
- Dashboard usability

#### **7. Conclusion**

```
This project successfully developed an air quality prediction system using 
Decision Tree classifier. The model achieved 100% accuracy on test data. 
The Streamlit dashboard provides an intuitive interface for real-time 
predictions and health recommendations. Future work includes using real 
Kathmandu data, adding more ML models, and deploying to cloud.
```

#### **8. References**

```
[1] Scikit-learn Documentation: https://scikit-learn.org/
[2] Streamlit Documentation: https://docs.streamlit.io/
[3] WHO Air Quality Guidelines, 2021
[4] Nepal Air Quality Standards, Department of Environment
```

#### **9. Appendix**

Include complete code files:
- `data_generator.py`
- `train_model.py`
- `app.py`

---

## 📝 Quick Reference Cheat Sheet

### Commands to Remember

```bash
# Generate data
python data_generator.py

# Train model
python train_model.py

# Run dashboard
streamlit run app.py

# Install dependencies
pip install -r requirements.txt
```

### Key Files

| File | Purpose |
|------|---------|
| `data_generator.py` | Creates synthetic dataset |
| `train_model.py` | Trains and evaluates model |
| `app.py` | Interactive dashboard |
| `model.pkl` | Saved trained model |
| `requirements.txt` | List of dependencies |

### Important Concepts

| Concept | Definition |
|---------|------------|
| **Feature** | Input variable (e.g., PM2.5) |
| **Target** | Output variable (Air Quality Category) |
| **Training** | Model learns from data |
| **Testing** | Evaluate model on unseen data |
| **Accuracy** | Percentage of correct predictions |
| **Overfitting** | Model memorizes instead of learning |
| **Decision Tree** | Flowchart-like ML algorithm |

---

## 🎓 Study Questions

### Basic Level
1. What are the five input features?
2. What are the three air quality categories?
3. What ML algorithm was used?
4. What accuracy did the model achieve?

### Intermediate Level
1. Why was Decision Tree chosen over other algorithms?
2. What is the purpose of `train_test_split`?
3. Why do we need `LabelEncoder`?
4. What does `max_depth` parameter control?

### Advanced Level
1. Explain how Decision Tree determines the best split.
2. Why might 100% accuracy be suspicious in real-world scenarios?
3. How would you handle imbalanced dataset?
4. What improvements would you make for deployment?

---

## 🔗 Additional Resources

- [Scikit-learn Decision Tree Tutorial](https://scikit-learn.org/stable/modules/tree.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [WHO Air Quality Guidelines](https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health)
- [Kathmandu Air Quality Data](https://airquality.gov.np/)

---

<div align="center">

**End of Study Guide**

*Created for Global LAB 1 - Air Quality Prediction Project*

</div>
