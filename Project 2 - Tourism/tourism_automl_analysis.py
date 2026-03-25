"""
Tourism AutoML Analysis
Automatically trains and compares multiple models for predicting tourist clusters
(Simplified version without MLJAR - uses sklearn models directly)
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_and_prepare_data(filepath):
    """
    Load clustered data and prepare for model training.
    """
    print("📂 Loading clustered dataset...")
    df = pd.read_csv(filepath)
    
    print(f"📊 Dataset shape: {df.shape}")
    
    # Encode Activity Preference
    encoder = LabelEncoder()
    df['Activity Preference Encoded'] = encoder.fit_transform(df['Activity Preference'])
    
    # Select features for prediction
    feature_columns = ['Age', 'Budget (NPR)', 'Duration (days)', 'Activity Preference Encoded', 'Spending Score']
    X = df[feature_columns]
    y = df['KMeans_Cluster']
    
    print(f"\n🔢 Features: {feature_columns}")
    print(f"🎯 Target: KMeans_Cluster (5 classes)")
    print(f"\n📈 Class distribution:")
    print(y.value_counts().sort_index())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✂️  Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, feature_columns, df


def train_multiple_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models automatically and compare performance.
    """
    print("\n" + "=" * 70)
    print("🤖 Training Multiple Models Automatically...")
    print("=" * 70)
    
    # Define models to try
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n🌳 Training {name}...")
        
        # Scale features for models that need it
        if name in ['K-Nearest Neighbors', 'SVM', 'Neural Network', 'Logistic Regression']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  ✅ {name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        results.append({
            'name': name,
            'model': model,
            'accuracy': accuracy,
            'y_pred': y_pred
        })
        
        trained_models[name] = model
    
    # Create leaderboard
    leaderboard = pd.DataFrame(results)[['name', 'accuracy']].sort_values('accuracy', ascending=False)
    leaderboard.columns = ['Model', 'Accuracy']
    
    print("\n" + "=" * 70)
    print("🏆 Model Leaderboard:")
    print("=" * 70)
    print(leaderboard.to_string(index=False))
    
    return results, leaderboard, trained_models


def evaluate_best_model(results, X_test, y_test):
    """
    Evaluate the best performing model in detail.
    """
    # Get best model
    best_result = results[0]  # Results already sorted by accuracy
    best_model = best_result['model']
    y_pred = best_result['y_pred']
    
    print("\n" + "=" * 70)
    print(f"📊 Detailed Evaluation - {best_result['name']}")
    print("=" * 70)
    
    accuracy = best_result['accuracy']
    print(f"\n✅ Best Model: {best_result['name']}")
    print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\n📈 Classification Report:")
    target_names = [f'Cluster {i}' for i in range(5)]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Leaderboard
    top_models = pd.DataFrame(results)[['name', 'accuracy']].nlargest(8, 'accuracy')
    axes[0].barh(range(len(top_models)), top_models['accuracy'], color='#3498db')
    axes[0].set_yticks(range(len(top_models)))
    axes[0].set_yticklabels(top_models['name'], fontsize=9)
    axes[0].set_xlabel('Accuracy', fontsize=12)
    axes[0].set_title('Top 8 Models Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_xlabel('Predicted Cluster', fontsize=12)
    axes[1].set_ylabel('True Cluster', fontsize=12)
    axes[1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Plot 3: Accuracy comparison chart
    all_accuracies = [r['accuracy'] for r in results]
    colors = ['#2ecc71' if r['accuracy'] == accuracy else '#3498db' for r in results]
    axes[2].bar(range(len(results)), all_accuracies, color=colors)
    axes[2].set_xticks(range(len(results)))
    axes[2].set_xticklabels([r['name'] for r in results], rotation=45, ha='right', fontsize=8)
    axes[2].set_ylabel('Accuracy', fontsize=12)
    axes[2].set_title('All Models Performance', fontsize=14, fontweight='bold')
    axes[2].axhline(y=max(all_accuracies), color='r', linestyle='--', label=f'Best: {max(all_accuracies):.3f}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('automl_model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n💾 Saved: automl_model_comparison.png")
    plt.close()
    
    # Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        feature_names = ['Age', 'Budget (NPR)', 'Duration (days)', 'Activity Preference', 'Spending Score']
        importance = best_model.feature_importances_
        
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        ax.barh(fi_df['Feature'], fi_df['Importance'], color='#2ecc71')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Feature Importance ({best_result["name"]})', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        for i, v in enumerate(fi_df['Importance']):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('automl_feature_importance.png', dpi=300, bbox_inches='tight')
        print("💾 Saved: automl_feature_importance.png")
        plt.close()
    
    return best_result


def save_automl_results(best_result, leaderboard, trained_models):
    """
    Save AutoML results.
    """
    print("\n💾 Saving AutoML results...")
    
    # Save best model
    joblib.dump(best_result['model'], 'automl_best_model.pkl')
    print("✅ Saved: automl_best_model.pkl")
    
    # Save all trained models
    joblib.dump(trained_models, 'all_trained_models.pkl')
    print("✅ Saved: all_trained_models.pkl")
    
    # Save evaluation results
    results_summary = {
        'best_model_name': best_result['name'],
        'best_accuracy': best_result['accuracy'],
        'leaderboard': leaderboard.to_dict()
    }
    joblib.dump(results_summary, 'automl_results_summary.pkl')
    print("✅ Saved: automl_results_summary.pkl")


def main():
    print("=" * 70)
    print("🏔️  Tourism AutoML Analysis")
    print("=" * 70)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_columns, df = \
        load_and_prepare_data(r"data\tourism_pokhara_clustered.csv")
    
    # Train multiple models
    results, leaderboard, trained_models = train_multiple_models(
        X_train, y_train, X_test, y_test
    )
    
    # Evaluate best model
    best_result = evaluate_best_model(results, X_test, y_test)
    
    # Save results
    save_automl_results(best_result, leaderboard, trained_models)
    
    print("\n" + "=" * 70)
    print("🎉 AutoML Analysis Complete!")
    print("=" * 70)
    print(f"\n📁 Generated files:")
    print("   - automl_best_model.pkl")
    print("   - all_trained_models.pkl")
    print("   - automl_results_summary.pkl")
    print("   - automl_model_comparison.png")
    print("   - automl_feature_importance.png")
    print("\n🚀 Next step: Run 'streamlit run tourism_app.py' to launch the dashboard!")


if __name__ == "__main__":
    main()
