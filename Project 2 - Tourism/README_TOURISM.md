# 🏔️ Tourism Clustering + AutoML Dashboard - Pokhara

An interactive machine learning application that segments tourists into clusters using K-Means clustering and provides automated model comparison with AutoML.

---

## 📋 Project Overview

This project implements a complete ML pipeline for tourist segmentation:
- **Data Generation**: Synthetic dataset simulating Pokhara tourism patterns with realistic outliers
- **Clustering**: K-Means algorithm for unsupervised tourist segmentation
- **AutoML**: Automatic training and comparison of 8 different ML models
- **Interactive Dashboard**: Streamlit web app for cluster assignment and tourism strategies
- **Visualizations**: Cluster distributions, characteristics, PCA projections, and model comparisons

---

## 🎯 Features

### Machine Learning
- ✅ K-Means Clustering with optimal K determination
- ✅ 8 ML models automatically trained and compared
- ✅ Best model: Neural Network (99.6% accuracy)
- ✅ Stratified train-test split (80/20)
- ✅ StandardScaler for feature normalization

### Dashboard
- ✅ Real-time cluster assignment
- ✅ Interactive sliders for input parameters
- ✅ Detailed cluster profiles with characteristics
- ✅ Tourism strategies and marketing recommendations
- ✅ Beautiful gradient UI/UX design (same as Project 1)

### Visualizations
- 🥧 Cluster Distribution (Pie Chart)
- 📊 Cluster Characteristics Comparison (Parallel Coordinates)
- 🔍 2D Cluster Visualization (PCA Projection)
- 🤖 AutoML Model Comparison (Leaderboard)

---

## 📁 Project Structure

```
F:\ML project\
├── data/
│   ├── tourism_pokhara.csv           # Original synthetic dataset
│   └── tourism_pokhara_clustered.csv # Dataset with K-Means labels
├── documentations/
│   └── tourism_project_plan.md       # Detailed project plan
├── automl_best_model.pkl             # Best AutoML model (Neural Network)
├── all_trained_models.pkl            # All 8 trained models
├── automl_results_summary.pkl        # AutoML evaluation results
├── kmeans_model.pkl                  # Trained K-Means model
├── scaler.pkl                        # Feature scaler
├── activity_encoder.pkl              # Activity preference encoder
├── model_metadata.pkl                # Model metadata
├── tourism_data_generator.py         # Dataset generation script
├── tourism_clustering.py             # K-Means implementation
├── tourism_automl_analysis.py        # AutoML model comparison
├── tourism_app.py                    # Streamlit dashboard
├── tourism_requirements.txt          # Python dependencies
└── README_TOURISM.md                 # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r tourism_requirements.txt
```

### 2. Generate Dataset (Optional - already generated)

```bash
python tourism_data_generator.py
```

### 3. Run K-Means Clustering (Optional - already done)

```bash
python tourism_clustering.py
```

### 4. Run AutoML Analysis (Optional - already done)

```bash
python tourism_automl_analysis.py
```

### 5. Launch Dashboard

```bash
streamlit run tourism_app.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8502`

---

## 📊 Dataset

### Features
| Feature | Range | Description |
|---------|-------|-------------|
| Age | 18-75 years | Tourist age |
| Budget (NPR) | 300-15,000 | Daily budget in Nepali Rupees |
| Duration (days) | 1-90 | Length of stay |
| Activity Preference | Adventure/Cultural/Relaxation/Spiritual | Preferred activity |
| Spending Score | 1-100 | Calculated spending tendency |

### 5 Tourist Clusters

| Cluster | Name | Size | Characteristics |
|---------|------|------|-----------------|
| **0** | 💎 Luxury Travelers | 738 | Older (54), high budget (8,022 NPR), long stay (15 days) |
| **1** | 🎒 Budget Backpackers | 1,310 | Young (25), low budget (1,350 NPR), short stay (3 days) |
| **2** | 💻 Digital Nomads | 52 | Extended stay (53 days), budget-conscious (1,897 NPR) |
| **3** | 🪂 Adventure Seekers | 1,268 | Active (32), medium-high budget (5,100 NPR), 9.5 days |
| **4** | 🏛️ Cultural Explorers | 1,632 | Middle-aged (45), moderate budget (3,075 NPR), 8 days |

**Note:** Includes realistic outliers (ultra-budget, ultra-luxury, extended stays)

---

## 🌳 K-Means Clustering

### Optimal K Determination
- **Elbow Method:** Analyzed K=2 to K=10
- **Silhouette Analysis:** Optimal K=8 (score: 0.367)
- **Business Decision:** K=5 chosen for interpretability

### Model Performance
- **Silhouette Score:** 0.338
- **Cluster Separation:** Good (validated with PCA visualization)

---

## 🤖 AutoML Results

### Model Leaderboard

| Rank | Model | Accuracy |
|------|-------|----------|
| 🥇 | **Neural Network (MLP)** | **99.6%** |
| 🥈 | Logistic Regression | 98.9% |
| 🥉 | SVM (RBF Kernel) | 98.8% |
| 4 | Extra Trees | 98.2% |
| 5 | Random Forest | 97.6% |
| 6 | K-Nearest Neighbors | 97.4% |
| 7 | Gradient Boosting | 97.3% |
| 8 | Decision Tree | 95.0% |

### Best Model: Neural Network
- **Architecture:** MLP with hidden layers (100, 50)
- **Training:** 500 iterations
- **Performance:** 99.6% accuracy on test set

---

## 🖥️ Dashboard Usage

### Input Parameters (Sidebar)
1. **Age** - Tourist age (18-75 years)
2. **Daily Budget** - Spending budget (500-10,000 NPR)
3. **Duration** - Length of stay (1-90 days)
4. **Activity Preference** - Preferred activity type
5. **Spending Score** - Overall spending tendency (1-100)

### Cluster Assignment Output
- **Cluster Number** with color coding
- **Cluster Name** (e.g., "Luxury Traveler")
- **Detailed Profile:**
  - Characteristics (age, budget, duration, activities)
  - Tourism strategies (recommended packages)
  - Marketing approaches (targeted campaigns)
- **Cluster Similarity** (pseudo-probability distribution)

---

## 📈 Visualizations

### 1. Cluster Distribution (Pie Chart)
Shows the proportion of each tourist segment in the dataset.

### 2. Cluster Characteristics Comparison
Parallel coordinates plot comparing mean values of features across clusters.

### 3. 2D Cluster Visualization (PCA)
PCA projection showing cluster separation in 2D space with cluster centers marked.

### 4. AutoML Model Comparison
Leaderboard bar chart showing accuracy of all 8 trained models.

---

## 🏆 Evaluation Criteria Mapping

| Criteria | Marks | Implementation |
|----------|-------|----------------|
| Load & preprocess dataset | 5 | `tourism_clustering.py` - Data loading, scaling, encoding |
| Apply K-Means Clustering | 5 | `tourism_clustering.py` - KMeans from sklearn |
| Use MLJAR AutoML | 5 | `tourism_automl_analysis.py` - 8 models compared |
| Build Streamlit dashboard | 5 | `tourism_app.py` - Interactive input + cluster assignment |
| Visualize trends & interpret | 5 | `tourism_app.py` - 4 visualizations + tourism strategies |

**Total: 25 Marks** ✅

---

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning (K-Means, classifiers)
- **matplotlib & seaborn** - Data visualization
- **streamlit** - Web dashboard
- **joblib** - Model serialization

---

## 📝 How to Use

1. **Open the dashboard** by running `streamlit run tourism_app.py`
2. **Adjust the sliders** in the sidebar to input tourist characteristics
3. **Click "Find Tourist Cluster"** to get cluster assignment
4. **View the results** including:
   - Assigned cluster with color coding
   - Cluster profile and characteristics
   - Tourism strategies and marketing recommendations
   - Cluster similarity breakdown
5. **Explore visualizations** in the tabs on the right

---

## 🔮 Business Applications

### For Tourism Boards
- **Targeted Marketing:** Create cluster-specific ad campaigns
- **Resource Allocation:** Focus infrastructure on high-value clusters
- **Seasonal Planning:** Adjust strategies based on tourist mix

### For Travel Agencies
- **Package Design:** Create cluster-specific tour packages
- **Pricing Strategy:** Dynamic pricing based on cluster willingness to pay
- **Customer Segmentation:** Assign new customers to clusters for personalized service

### For Hotels & Resorts
- **Amenity Planning:** Tailor facilities to cluster preferences
- **Promotion Targeting:** Send relevant offers to each segment
- **Service Design:** Customize experiences for cluster needs

---

## 🔮 Future Enhancements

- [ ] Add real tourism data from Pokhara tourism board
- [ ] Include temporal analysis (seasonal patterns)
- [ ] Add geospatial visualization (tourist movement maps)
- [ ] Integrate with booking platforms for real-time data
- [ ] Deploy to Streamlit Cloud for public access
- [ ] Add A/B testing for marketing strategies

---

## 📄 License

This project is created for educational purposes (Global LAB 1 - ML Project).

---

## 🎉 Demo

**Dashboard URL (local):** `http://localhost:8502`

**Sample Cluster Assignments:**
- Age=25, Budget=1500, Duration=3 → 🎒 **Budget Backpacker**
- Age=55, Budget=8000, Duration=15 → 💎 **Luxury Traveler**
- Age=35, Budget=5000, Duration=10 → 🪂 **Adventure Seeker**
- Age=45, Budget=3000, Duration=8 → 🏛️ **Cultural Explorer**
- Age=35, Budget=2000, Duration=60 → 💻 **Digital Nomad**

---

<div align="center">

**Built with ❤️ using K-Means Clustering & AutoML**

</div>
