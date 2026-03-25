"""
🌬️ Air Quality Prediction Dashboard
Interactive Streamlit app for real-time air quality prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# Get the directory where the script is located
BASE_DIR = Path(__file__).parent

# Page configuration
st.set_page_config(
    page_title="Air Quality Prediction Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load trained model and artifacts."""
    model = joblib.load(BASE_DIR / 'models' / 'model.pkl')
    label_encoder = joblib.load(BASE_DIR / 'models' / 'label_encoder.pkl')
    metadata = joblib.load(BASE_DIR / 'models' / 'model_metadata.pkl')
    return model, label_encoder, metadata


@st.cache_data
def load_dataset():
    """Load the air quality dataset."""
    df = pd.read_csv(BASE_DIR / 'data' / 'air_quality_kathmandu.csv')
    return df


# Custom CSS for beautiful UI/UX
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

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Prediction result box - Good (Green) */
    .prediction-good {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        border-radius: 15px;
        padding: 30px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Prediction result box - Moderate (Orange/Amber) */
    .prediction-moderate {
        background: linear-gradient(135deg, #f39c12 0%, #f1c40f 100%);
        border-radius: 15px;
        padding: 30px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Prediction result box - Unhealthy (Red) */
    .prediction-unhealthy {
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
def load_model():
    """Load trained model and artifacts."""
    model = joblib.load('models/model.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    metadata = joblib.load('models/model_metadata.pkl')
    return model, label_encoder, metadata


@st.cache_data
def load_dataset():
    """Load the air quality dataset."""
    df = pd.read_csv('data/air_quality_kathmandu.csv')
    return df


def get_health_recommendation(category):
    """Return health recommendations based on air quality category."""
    recommendations = {
        'Good': {
            'title': '✅ Air Quality is Good',
            'message': 'Perfect day to be outdoors! Enjoy activities outside.',
            'tips': [
                '🏃 Ideal for outdoor exercise',
                '🌳 Enjoy nature walks',
                '🚴 Great for cycling',
                '😊 No health concerns'
            ]
        },
        'Moderate': {
            'title': '⚠️ Air Quality is Moderate',
            'message': 'Sensitive individuals should consider limiting prolonged outdoor exposure.',
            'tips': [
                '👶 Sensitive groups should limit outdoor time',
                '🏃‍♂️ Reduce intense outdoor activities if symptomatic',
                '😷 Consider wearing a mask in high-traffic areas',
                '🏠 Keep windows open for ventilation'
            ]
        },
        'Unhealthy': {
            'title': '🚨 Air Quality is Unhealthy',
            'message': 'Everyone should reduce prolonged outdoor exertion.',
            'tips': [
                '😷 Wear N95 masks when going outside',
                '🏠 Stay indoors as much as possible',
                '🪟 Keep windows and doors closed',
                '💨 Use air purifiers if available',
                '👴 Elderly and children should avoid outdoors'
            ]
        }
    }
    return recommendations.get(category, recommendations['Moderate'])


def plot_correlation_heatmap(df, feature_columns):
    """Create correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[feature_columns].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_columns):
    """Create feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    importance = model.feature_importances_
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax.barh(feature_columns, importance, color=colors[:len(feature_columns)])
    
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance (Decision Tree)', fontsize=16, fontweight='bold', pad=20)
    ax.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, importance):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim(right=max(importance) * 1.2)
    plt.tight_layout()
    return fig


def plot_air_quality_distribution(df):
    """Create pie chart for air quality distribution."""
    fig, ax = plt.subplots(figsize=(8, 8))

    category_counts = df['Air Quality Category'].value_counts()
    # Updated colors: Green, Amber/Orange, Red
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    wedges, texts, autotexts = ax.pie(
        category_counts.values,
        labels=category_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.05] * len(category_counts),
        shadow=True,
        startangle=90
    )

    # Style the text
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
        autotext.set_color('white')

    ax.set_title('Air Quality Category Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


def plot_pm_scatter(df):
    """Create PM2.5 vs PM10 scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Map categories to colors (Updated: Green, Amber, Red)
    category_colors = {
        'Good': '#2ecc71',
        'Moderate': '#f39c12',
        'Unhealthy': '#e74c3c'
    }

    for category in df['Air Quality Category'].unique():
        subset = df[df['Air Quality Category'] == category]
        ax.scatter(
            subset['PM2.5'],
            subset['PM10'],
            c=category_colors[category],
            label=category,
            alpha=0.6,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )

    ax.set_xlabel('PM2.5 (µg/m³)', fontsize=12, fontweight='bold')
    ax.set_ylabel('PM10 (µg/m³)', fontsize=12, fontweight='bold')
    ax.set_title('PM2.5 vs PM10 by Air Quality Category', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    # Load model and data
    try:
        model, label_encoder, metadata = load_model()
        df = load_dataset()
    except Exception as e:
        st.error(f"❌ Error loading model or data: {e}")
        st.stop()
    
    feature_columns = metadata['feature_columns']

    # Header
    st.markdown("""
        <div class="main-header">
            <h1 style="color: #2c3e50; margin: 0;">🌬️ Air Quality Prediction Dashboard</h1>
            <p style="color: #555; margin: 10px 0 0 0; font-size: 16px;">
                Kathmandu Pollution Monitoring • Powered by Machine Learning
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
            <p style="font-size: 18px; font-weight: bold; margin: 5px 0; color: #ffffff;">
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
        
        st.markdown("### 📝 Input Parameters")
        st.markdown("Enter the air quality measurements below:")
        
        pm25 = st.slider(
            "🌫️ PM2.5 (µg/m³)",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=1.0,
            help="Fine particulate matter (diameter < 2.5µm)"
        )
        
        pm10 = st.slider(
            "💨 PM10 (µg/m³)",
            min_value=0.0,
            max_value=600.0,
            value=100.0,
            step=1.0,
            help="Coarse particulate matter (diameter < 10µm)"
        )
        
        temperature = st.slider(
            "🌡️ Temperature (°C)",
            min_value=5.0,
            max_value=40.0,
            value=25.0,
            step=0.5,
            help="Ambient air temperature"
        )
        
        humidity = st.slider(
            "💧 Humidity (%)",
            min_value=20.0,
            max_value=95.0,
            value=60.0,
            step=1.0,
            help="Relative humidity percentage"
        )
        
        wind_speed = st.slider(
            "💨 Wind Speed (km/h)",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=0.5,
            help="Wind velocity"
        )
        
        st.markdown("---")
        
        # Predict button
        predict_btn = st.button("🔮 Predict Air Quality", use_container_width=True)
        
        # Dataset info
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Features", len(feature_columns))
        st.metric("Model Accuracy", f"{metadata['accuracy']*100:.1f}%")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔮 Prediction Result")
        
        if predict_btn:
            # Prepare input
            input_data = np.array([[pm25, pm10, temperature, humidity, wind_speed]])
            
            # Predict
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Decode prediction
            predicted_category = label_encoder.inverse_transform([prediction])[0]
            
            # Get recommendation
            recommendation = get_health_recommendation(predicted_category)
            
            # Display prediction with styled box
            if predicted_category == 'Good':
                css_class = 'prediction-good'
            elif predicted_category == 'Moderate':
                css_class = 'prediction-moderate'
            else:
                css_class = 'prediction-unhealthy'
            
            st.markdown(f"""
                <div class="{css_class}">
                    <div style="font-size: 48px; margin-bottom: 10px;">
                        {'🟢' if predicted_category == 'Good' else '🟡' if predicted_category == 'Moderate' else '🔴'}
                    </div>
                    <div>{predicted_category}</div>
                    <div style="font-size: 16px; margin-top: 10px; opacity: 0.9;">
                        Confidence: {max(prediction_proba)*100:.1f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Display health recommendations
            st.markdown(f"### {recommendation['title']}")
            st.info(recommendation['message'], icon="ℹ️")
            
            st.markdown("**Health Tips:**")
            for tip in recommendation['tips']:
                st.markdown(f"- {tip}")
            
            # Prediction confidence breakdown
            st.markdown("---")
            st.markdown("### 📈 Prediction Confidence")

            confidence_df = pd.DataFrame({
                'Category': label_encoder.classes_,
                'Confidence': prediction_proba * 100
            })

            st.bar_chart(
                confidence_df.set_index('Category'),
                color='#2c3e50',
                use_container_width=True
            )
        
        else:
            st.info("👈 Adjust the sliders and click **Predict** to see results!", icon="👈")
            
            # Show sample prediction
            st.markdown("### 📋 How it works:")
            st.markdown("""
            1. **Input** your air quality measurements using the sliders
            2. **Click Predict** to get instant results
            3. **View** the predicted air quality category
            4. **Read** health recommendations and tips
            """)
    
    with col2:
        st.markdown("### 📊 Data Visualizations")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔥 Correlation",
            "📊 Feature Importance", 
            "🥧 Distribution",
            "⚡ PM Scatter"
        ])
        
        with tab1:
            st.markdown("#### Feature Correlation Heatmap")
            fig = plot_correlation_heatmap(df, feature_columns)
            st.pyplot(fig, use_container_width=True)
            st.caption("Shows how features correlate with each other. Warmer colors indicate stronger positive correlation.")
        
        with tab2:
            st.markdown("#### Feature Importance")
            fig = plot_feature_importance(model, feature_columns)
            st.pyplot(fig, use_container_width=True)
            st.caption("Shows which features are most important for the model's predictions.")
        
        with tab3:
            st.markdown("#### Air Quality Distribution")
            fig = plot_air_quality_distribution(df)
            st.pyplot(fig, use_container_width=True)
            st.caption("Distribution of air quality categories in the Kathmandu dataset.")
        
        with tab4:
            st.markdown("#### PM2.5 vs PM10 Scatter Plot")
            fig = plot_pm_scatter(df)
            st.pyplot(fig, use_container_width=True)
            st.caption("Relationship between PM2.5 and PM10 levels, colored by air quality category.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: white; padding: 20px;">
            <p style="margin: 0;">
                🌍 <strong>Air Quality Prediction Dashboard</strong> • Built with Streamlit & Machine Learning
            </p>
            <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.8;">
                Data Source: Kathmandu Pollution Monitoring Stations • Model: Decision Tree Classifier
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
