"""
Air Quality Prediction Model Training
Trains a Decision Tree classifier and evaluates performance
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    ConfusionMatrixDisplay
)


def load_and_preprocess_data(filepath):
    """
    Load dataset and preprocess for training.
    
    Returns:
        X_train, X_test, y_train, y_test, label_encoder
    """
    print("📂 Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"\n📋 Columns: {list(df.columns)}")
    print(f"\n📈 Target distribution:")
    print(df['Air Quality Category'].value_counts())
    
    # Separate features and target
    feature_columns = ['PM2.5', 'PM10', 'Temperature', 'Humidity', 'Wind Speed']
    X = df[feature_columns]
    y = df['Air Quality Category']
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\n🔢 Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Split data (80% train, 20% test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_encoded
    )
    
    print(f"\n✂️  Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, label_encoder, feature_columns


def train_decision_tree(X_train, y_train, random_state=42):
    """
    Train a Decision Tree classifier.
    """
    print("\n🌳 Training Decision Tree Classifier...")
    
    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    print("✅ Model training complete!")
    
    return model


def evaluate_model(model, X_test, y_test, label_encoder, feature_columns):
    """
    Evaluate model performance and generate visualizations.
    """
    print("\n📊 Evaluating model performance...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📋 Confusion Matrix:")
    print(cm)
    
    # Classification Report
    print(f"\n📈 Classification Report:")
    target_names = label_encoder.classes_
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Generate visualizations
    print("\n📊 Generating evaluation plots...")
    
    # 1. Confusion Matrix Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names, 
                ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: confusion_matrix.png")
    plt.close()
    
    # 2. Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    importance = model.feature_importances_
    bars = ax.barh(feature_columns, importance, color='#2E86AB')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance (Decision Tree)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels on bars
    for bar, val in zip(bars, importance):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: feature_importance.png")
    plt.close()
    
    return accuracy, cm


def save_model_and_artifacts(model, label_encoder, feature_columns, accuracy):
    """
    Save trained model, label encoder, and metadata.
    """
    print("\n💾 Saving model and artifacts...")
    
    # Save model
    joblib.dump(model, 'model.pkl')
    print("✅ Saved: model.pkl")
    
    # Save label encoder
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("✅ Saved: label_encoder.pkl")
    
    # Save metadata
    metadata = {
        'feature_columns': feature_columns,
        'accuracy': accuracy,
        'classes': list(label_encoder.classes_)
    }
    joblib.dump(metadata, 'model_metadata.pkl')
    print("✅ Saved: model_metadata.pkl")


def main():
    print("=" * 60)
    print("🌬️  Air Quality Prediction - Model Training")
    print("=" * 60)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoder, feature_columns = \
        load_and_preprocess_data(r"data\air_quality_kathmandu.csv")
    
    # Train model
    model = train_decision_tree(X_train, y_train)
    
    # Evaluate model
    accuracy, cm = evaluate_model(model, X_test, y_test, label_encoder, feature_columns)
    
    # Save model and artifacts
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


if __name__ == "__main__":
    main()
