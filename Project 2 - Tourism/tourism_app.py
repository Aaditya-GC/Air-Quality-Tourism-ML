"""
🏔️ Tourism Clustering Dashboard - Pokhara
Interactive Streamlit app for tourist cluster analysis and prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Get the directory where the script is located
BASE_DIR = Path(__file__).parent

# Page configuration
st.set_page_config(
    page_title="Tourism Clustering Dashboard - Pokhara",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    """Load trained models and artifacts."""
    kmeans = joblib.load(BASE_DIR / 'models' / 'kmeans_model.pkl')
    scaler = joblib.load(BASE_DIR / 'models' / 'scaler.pkl')
    encoder = joblib.load(BASE_DIR / 'models' / 'activity_encoder.pkl')
    metadata = joblib.load(BASE_DIR / 'models' / 'model_metadata.pkl')

    # Load AutoML best model
    try:
        automl_model = joblib.load(BASE_DIR / 'models' / 'automl_best_model.pkl')
    except:
        automl_model = None

    return kmeans, scaler, encoder, metadata, automl_model


@st.cache_data
def load_dataset():
    """Load the tourism dataset."""
    df = pd.read_csv(BASE_DIR / 'data' / 'tourism_pokhara_clustered.csv')
    return df


# Custom CSS for beautiful UI/UX (same style as Project 1)
st.markdown("""
<style>
    /* Main background - Clean light theme */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Global text color - Dark for readability */
    .stApp {
        color: #2c3e50;
    }

    /* Card styling */
    .css-1r6slb0 {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 20px;
    }

    /* Sidebar text */
    .css-1d391kg * {
        color: #2c3e50 !important;
    }

    /* Cluster result boxes */
    .cluster-0 {
        background: linear-gradient(135deg, #8e44ad 0%, #9b59b6 100%);
        border-radius: 15px;
        padding: 30px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .cluster-1 {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        border-radius: 15px;
        padding: 30px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .cluster-2 {
        background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%);
        border-radius: 15px;
        padding: 30px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .cluster-3 {
        background: linear-gradient(135deg, #2980b9 0%, #3498db 100%);
        border-radius: 15px;
        padding: 30px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .cluster-4 {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        border-radius: 15px;
        padding: 30px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Header */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Headings - Dark color */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
        font-weight: 700;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }

    /* Slider styling */
    .stSlider>div>div {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
    }

    /* Labels - Dark color */
    label, .stMarkdown, .st-ae, .st-ag {
        color: #2c3e50 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models and artifacts."""
    kmeans = joblib.load('models/kmeans_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    encoder = joblib.load('models/activity_encoder.pkl')
    metadata = joblib.load('models/model_metadata.pkl')

    # Load AutoML best model
    try:
        automl_model = joblib.load('models/automl_best_model.pkl')
    except:
        automl_model = None

    return kmeans, scaler, encoder, metadata, automl_model


@st.cache_data
def load_dataset():
    """Load the tourism dataset."""
    df = pd.read_csv('data/tourism_pokhara_clustered.csv')
    return df


def get_cluster_profile(cluster_num):
    """Return detailed cluster profile and tourism strategies."""
    profiles = {
        0: {
            'name': '💎 Luxury Travelers',
            'description': 'Older, affluent tourists who prefer extended stays in premium accommodations.',
            'characteristics': [
                '👴 Age: 50-65 years (mature travelers)',
                '💰 Budget: 7,000-9,000 NPR/day (high spending)',
                '📅 Duration: 14-18 days (extended vacation)',
                '🎯 Activities: Relaxation, Cultural experiences',
                '⭐ Spending Score: 35-40 (premium spenders)'
            ],
            'strategies': [
                '🏨 Partner with 5-star hotels and resorts',
                '🍷 Offer wine tasting and fine dining packages',
                '🧘 Create wellness and spa retreat packages',
                '🚁 Helicopter tours of Himalayas',
                '👨‍👩‍👧 Private guided cultural tours',
                '💆 Luxury spa and wellness packages'
            ],
            'marketing': [
                'Target through luxury travel magazines',
                'Partner with premium travel agencies',
                'Offer exclusive VIP experiences',
                'Create loyalty programs for repeat visitors'
            ]
        },
        1: {
            'name': '🎒 Budget Backpackers',
            'description': 'Young travelers exploring Pokhara on a tight budget with short stays.',
            'characteristics': [
                '🧑 Age: 20-30 years (young adults)',
                '💰 Budget: 1,000-1,800 NPR/day (budget-conscious)',
                '📅 Duration: 2-5 days (short trip)',
                '🎯 Activities: Adventure, budget sightseeing',
                '⭐ Spending Score: 5-10 (minimal spenders)'
            ],
            'strategies': [
                '🏠 Promote hostels and budget accommodations',
                '🍜 Street food tours and local eateries',
                '🚌 Group transportation discounts',
                '🎫 Budget adventure activity packages',
                '🗺️ Free walking tour maps',
                '📱 Social media marketing campaigns'
            ],
            'marketing': [
                'Instagram and TikTok campaigns',
                'Partner with hostel booking platforms',
                'Create backpacker discount cards',
                'Influencer collaborations'
            ]
        },
        2: {
            'name': '💻 Digital Nomads',
            'description': 'Extended-stay visitors who work remotely while exploring Pokhara.',
            'characteristics': [
                '🧑 Age: 30-40 years (working professionals)',
                '💰 Budget: 1,500-2,500 NPR/day (moderate)',
                '📅 Duration: 45-70 days (long-term stay)',
                '🎯 Activities: Work-leisure balance',
                '⭐ Spending Score: 30-45 (sustained spending)'
            ],
            'strategies': [
                '📶 Co-working space memberships',
                '🏠 Monthly apartment rentals',
                '☕ Cafe hopping guides with WiFi',
                '🧘 Weekend wellness retreats',
                '🤝 Networking events for expats',
                '📅 Long-stay discount packages'
            ],
            'marketing': [
                'Nomad listing platforms (Nomad List)',
                'Remote work community groups',
                'Extended stay promotions',
                'Testimonials from current nomads'
            ]
        },
        3: {
            'name': '🪂 Adventure Seekers',
            'description': 'Active tourists seeking thrilling experiences and outdoor adventures.',
            'characteristics': [
                '🧑 Age: 28-38 years (active adults)',
                '💰 Budget: 4,500-5,500 NPR/day (moderate-high)',
                '📅 Duration: 8-12 days (activity-filled)',
                '🎯 Activities: Paragliding, Trekking, Rafting',
                '⭐ Spending Score: 20-28 (experience spenders)'
            ],
            'strategies': [
                '🪂 Paragliding package deals',
                '🥾 Trekking guide services',
                '🚣 White water rafting tours',
                '🚵 Mountain biking rentals',
                '🧗 Rock climbing expeditions',
                '📸 Adventure photography tours'
            ],
            'marketing': [
                'Adventure travel blogs and websites',
                'YouTube adventure vlogs',
                'Partnership with adventure gear brands',
                'Action sports event sponsorships'
            ]
        },
        4: {
            'name': '🏛️ Cultural Explorers',
            'description': 'Middle-aged tourists interested in heritage, traditions, and local experiences.',
            'characteristics': [
                '🧑 Age: 40-55 years (mature adults)',
                '💰 Budget: 2,800-3,500 NPR/day (moderate)',
                '📅 Duration: 6-10 days (cultural immersion)',
                '🎯 Activities: Temples, Museums, Local villages',
                '⭐ Spending Score: 12-18 (cultural spenders)'
            ],
            'strategies': [
                '🏛️ Heritage site circuit packages',
                '🎭 Traditional cultural show tickets',
                '🍳 Cooking classes (Nepali cuisine)',
                '🏘️ Village homestay experiences',
                '🎨 Handicraft workshop visits',
                '📚 Historical guided tours'
            ],
            'marketing': [
                'Cultural travel magazines',
                'UNESCO heritage tourism platforms',
                'Documentary-style promotional videos',
                'Partnership with cultural organizations'
            ]
        }
    }
    return profiles.get(cluster_num, profiles[4])


def plot_cluster_distribution(df):
    """Create pie chart for cluster distribution."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    cluster_counts = df['KMeans_Cluster'].value_counts().sort_index()
    colors = ['#8e44ad', '#2ecc71', '#f39c12', '#3498db', '#e74c3c']
    
    wedges, texts, autotexts = ax.pie(
        cluster_counts.values,
        labels=[f'Cluster {i}' for i in cluster_counts.index],
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.05] * len(cluster_counts),
        shadow=True,
        startangle=90
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    ax.set_title('Tourist Cluster Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


def plot_cluster_characteristics(df):
    """Create parallel coordinates plot for cluster characteristics."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    numeric_features = ['Age', 'Budget (NPR)', 'Duration (days)', 'Spending Score']
    cluster_means = df.groupby('KMeans_Cluster')[numeric_features].mean()
    
    colors = ['#8e44ad', '#2ecc71', '#f39c12', '#3498db', '#e74c3c']
    
    for idx, cluster in cluster_means.iterrows():
        ax.plot(cluster_means.columns, cluster.values, marker='o',
                linewidth=2, markersize=8, label=f'Cluster {idx}',
                color=colors[idx % len(colors)])
    
    ax.set_title('Cluster Characteristics Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Mean Values', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_2d_clusters(df, kmeans):
    """Create 2D scatter plot of clusters."""
    from sklearn.decomposition import PCA
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    numeric_features = ['Age', 'Budget (NPR)', 'Duration (days)', 'Spending Score']
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df[numeric_features])
    
    colors = ['#8e44ad', '#2ecc71', '#f39c12', '#3498db', '#e74c3c']
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['KMeans_Cluster'],
                         cmap='tab10', alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # Add cluster centers
    cluster_centers_numeric = kmeans.cluster_centers_[:, :4]
    centers_pca = pca.transform(cluster_centers_numeric)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200,
               marker='X', edgecolors='black', linewidth=2, label='Cluster Centers')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('Tourist Clusters (PCA 2D Projection)', fontsize=16, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_automl_results():
    """Create AutoML model comparison chart."""
    try:
        results = joblib.load('automl_results_summary.pkl')
        leaderboard = pd.DataFrame(results['leaderboard'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        top_models = leaderboard.nlargest(8, 'Accuracy')
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(top_models))]
        
        ax.barh(range(len(top_models)), top_models['Accuracy'], color=colors)
        ax.set_yticks(range(len(top_models)))
        ax.set_yticklabels(top_models['Model'], fontsize=10)
        ax.set_xlabel('Accuracy', fontsize=12)
        ax.set_title('AutoML Model Performance', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, v in enumerate(top_models['Accuracy']):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        return None


def main():
    # Load models and data
    try:
        kmeans, scaler, encoder, metadata, automl_model = load_models()
        df = load_dataset()
    except Exception as e:
        st.error(f"❌ Error loading models or data: {e}")
        st.stop()
    
    feature_columns = metadata['feature_columns']
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1 style="color: #2c3e50; margin: 0;">🏔️ Tourism Clustering Dashboard</h1>
            <p style="color: #555; margin: 10px 0 0 0; font-size: 16px;">
                Pokhara Tourist Segmentation • Powered by K-Means & AutoML
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for user input
    with st.sidebar:
        # Profile section with photo
        st.markdown("### 👤 Prepared By")

        # Use relative path to photo in parent folder
        st.image("../photo.JPG", width=150)

        # Your name and details
        st.markdown("""
        <div style="text-align: center; margin-top: 10px;">
            <p style="font-size: 18px; font-weight: bold; margin: 5px 0; color: #FFFFFF;">
                Aaditya GC
            </p>
            <p style="font-size: 14px; color: #666; margin: 3px 0;">
                🎓 Student
            </p>
            <p style="font-size: 13px; color: #888; margin: 3px 0;">
                Roll No: 2
            </p>
            <p style="font-size: 13px; color: #888; margin: 3px 0;">
                📅 March 2026
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📝 Tourist Profile")
        st.markdown("Enter tourist characteristics:")
        
        age = st.slider(
            "🎂 Age (years)",
            min_value=18,
            max_value=75,
            value=30,
            step=1,
            help="Tourist age"
        )
        
        budget = st.slider(
            "💰 Daily Budget (NPR)",
            min_value=500,
            max_value=10000,
            value=3000,
            step=100,
            help="Daily spending budget in Nepali Rupees"
        )
        
        duration = st.slider(
            "📅 Duration (days)",
            min_value=1,
            max_value=90,
            value=7,
            step=1,
            help="Length of stay in days"
        )
        
        activity = st.selectbox(
            "🎯 Activity Preference",
            options=['Adventure', 'Cultural', 'Relaxation', 'Spiritual'],
            help="Preferred activity type"
        )
        
        spending_score = st.slider(
            "⭐ Spending Score",
            min_value=1,
            max_value=100,
            value=20,
            step=1,
            help="Overall spending tendency (1-100)"
        )
        
        st.markdown("---")
        
        # Predict button
        predict_btn = st.button("🔍 Find Tourist Cluster", use_container_width=True)
        
        # Dataset info
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.metric("Total Tourists", f"{len(df):,}")
        st.metric("Clusters", "5")
        if automl_model:
            st.metric("AutoML Best Accuracy", "99.6%")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Cluster Assignment")
        
        if predict_btn:
            # Prepare input
            activity_encoded = encoder.transform([activity])[0]
            
            input_data = np.array([[age, budget, duration, activity_encoded, spending_score]])
            input_scaled = scaler.transform(input_data)
            
            # Predict cluster using K-Means
            cluster = kmeans.predict(input_scaled)[0]
            
            # Get cluster profile
            profile = get_cluster_profile(cluster)
            
            # Display cluster result with styled box
            st.markdown(f"""
                <div class="cluster-{cluster}">
                    <div style="font-size: 48px; margin-bottom: 10px;">
                        {['💎', '🎒', '💻', '🪂', '🏛️'][cluster]}
                    </div>
                    <div>Cluster {cluster}</div>
                    <div style="font-size: 18px; margin-top: 10px; opacity: 0.9;">
                        {profile['name']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Display cluster description
            st.markdown(f"### {profile['name']}")
            st.info(profile['description'], icon="ℹ️")
            
            # Display characteristics
            st.markdown("**📋 Characteristics:**")
            for char in profile['characteristics']:
                st.markdown(f"- {char}")
            
            # Display tourism strategies
            st.markdown("---")
            st.markdown("### 🎯 Tourism Strategies")
            
            with st.expander("📦 Recommended Packages", expanded=True):
                for strategy in profile['strategies']:
                    st.markdown(f"- {strategy}")
            
            with st.expander("📢 Marketing Approaches"):
                for marketing in profile['marketing']:
                    st.markdown(f"- {marketing}")
            
            # Cluster assignment confidence
            st.markdown("---")
            st.markdown("### 📊 Cluster Assignment Details")
            
            # Calculate distance to cluster centers
            distances = kmeans.transform(input_scaled)[0]
            probabilities = 1 / (1 + distances)  # Convert to pseudo-probabilities
            probabilities = probabilities / probabilities.sum() * 100
            
            confidence_df = pd.DataFrame({
                'Cluster': [f'Cluster {i}' for i in range(5)],
                'Similarity': probabilities
            })
            
            st.bar_chart(
                confidence_df.set_index('Cluster'),
                color='#2c3e50',
                use_container_width=True
            )
            
        else:
            st.info("👈 Enter tourist profile and click **Find Tourist Cluster** to see results!", icon="👈")
            
            # Show sample info
            st.markdown("### 📋 How it works:")
            st.markdown("""
            1. **Input** tourist characteristics using the sliders
            2. **Click Find Cluster** to get segment assignment
            3. **View** the tourist cluster profile
            4. **Read** targeted strategies and marketing approaches
            """)
    
    with col2:
        st.markdown("### 📊 Cluster Visualizations")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "🥧 Distribution",
            "📊 Characteristics",
            "🔍 2D Clusters",
            "🤖 AutoML Results"
        ])
        
        with tab1:
            st.markdown("#### Tourist Cluster Distribution")
            fig = plot_cluster_distribution(df)
            st.pyplot(fig, use_container_width=True)
            st.caption("Distribution of tourists across 5 segments in Pokhara.")
        
        with tab2:
            st.markdown("#### Cluster Characteristics Comparison")
            fig = plot_cluster_characteristics(df)
            st.pyplot(fig, use_container_width=True)
            st.caption("Comparison of average features across clusters.")
        
        with tab3:
            st.markdown("#### 2D Cluster Visualization (PCA)")
            fig = plot_2d_clusters(df, kmeans)
            st.pyplot(fig, use_container_width=True)
            st.caption("PCA projection showing cluster separation in 2D space.")
        
        with tab4:
            st.markdown("#### AutoML Model Comparison")
            fig = plot_automl_results()
            if fig:
                st.pyplot(fig, use_container_width=True)
                st.caption("Performance comparison of automatically trained models for cluster prediction.")
            else:
                st.info("AutoML results not available.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #555; padding: 20px;">
            <p style="margin: 0;">
                🏔️ <strong>Tourism Clustering Dashboard</strong> • Built with K-Means Clustering & AutoML
            </p>
            <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.8;">
                Data: Pokhara Tourism • Models: K-Means (5 clusters) + Neural Network (99.6% accuracy)
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
