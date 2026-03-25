"""
Tourism Clustering with K-Means
Applies K-Means clustering to segment tourists and saves the model
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score


def load_and_preprocess_data(filepath):
    """
    Load dataset and preprocess for clustering.
    
    Returns:
        X_scaled: Scaled features
        df: Original dataframe
        scaler: Fitted StandardScaler
        encoder: Fitted LabelEncoder for Activity Preference
        feature_columns: List of feature names
    """
    print("📂 Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"\n📋 Columns: {list(df.columns)}")
    
    # Store original cluster labels (ground truth)
    original_clusters = df['Cluster'].copy()
    
    # Drop original cluster column for unsupervised learning
    df_clustering = df.drop('Cluster', axis=1)
    
    # Encode categorical variable (Activity Preference)
    encoder = LabelEncoder()
    df_clustering['Activity Preference'] = encoder.fit_transform(df_clustering['Activity Preference'])
    
    print(f"\n🔢 Activity Preference encoding: {dict(zip(encoder.classes_, range(len(encoder.classes_))))}")
    
    # Select features for clustering
    feature_columns = ['Age', 'Budget (NPR)', 'Duration (days)', 'Activity Preference', 'Spending Score']
    X = df_clustering[feature_columns]
    
    # Scale features (IMPORTANT for K-Means!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n✂️  Features: {feature_columns}")
    print(f"📏 Scaling: StandardScaler applied")
    
    return X_scaled, df, scaler, encoder, feature_columns, original_clusters


def find_optimal_k(X_scaled, max_k=10):
    """
    Find optimal number of clusters using Elbow Method and Silhouette Analysis.
    """
    print("\n🔍 Finding optimal number of clusters...")
    
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        calinski_scores.append(calinski_harabasz_score(X_scaled, labels))
        davies_scores.append(davies_bouldin_score(X_scaled, labels))
        
        print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
    
    # Find optimal K (maximum silhouette score)
    optimal_k_idx = np.argmax(silhouette_scores)
    optimal_k = list(k_range)[optimal_k_idx]
    
    print(f"\n📊 Optimal K (by Silhouette): {optimal_k} (score: {silhouette_scores[optimal_k_idx]:.3f})")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Elbow Method
    axes[0, 0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[0, 0].set_ylabel('Inertia', fontsize=12)
    axes[0, 0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Silhouette Score
    axes[0, 1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[0, 1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[0, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[0, 1].set_ylabel('Silhouette Score', fontsize=12)
    axes[0, 1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Calinski-Harabasz Score
    axes[1, 0].plot(k_range, calinski_scores, 'ro-', linewidth=2, markersize=8)
    axes[1, 0].axvline(x=optimal_k, color='g', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[1, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[1, 0].set_ylabel('Calinski-Harabasz Score', fontsize=12)
    axes[1, 0].set_title('Calinski-Harabasz Index', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Davies-Bouldin Index (lower is better)
    axes[1, 1].plot(k_range, davies_scores, 'mo-', linewidth=2, markersize=8)
    axes[1, 1].axvline(x=optimal_k, color='g', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[1, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[1, 1].set_ylabel('Davies-Bouldin Index', fontsize=12)
    axes[1, 1].set_title('Davies-Bouldin Index', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimal_k_analysis.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: optimal_k_analysis.png")
    plt.close()
    
    return optimal_k


def apply_kmeans(X_scaled, df, n_clusters=5, random_state=42):
    """
    Apply K-Means clustering and analyze results.
    """
    print(f"\n🎯 Applying K-Means Clustering with K={n_clusters}...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=random_state
    )
    
    # Fit and predict
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df['KMeans_Cluster'] = cluster_labels
    
    print("✅ K-Means clustering complete!")
    
    return kmeans, df


def analyze_clusters(X_scaled, df, kmeans, feature_columns):
    """
    Analyze and visualize cluster characteristics.
    """
    print("\n📊 Analyzing cluster characteristics...")
    
    # Cluster statistics
    cluster_stats = df.groupby('KMeans_Cluster').agg({
        'Age': ['mean', 'median', 'std'],
        'Budget (NPR)': ['mean', 'median', 'std'],
        'Duration (days)': ['mean', 'median', 'std'],
        'Spending Score': ['mean', 'median', 'std']
    }).round(2)
    
    print(f"\n📈 Cluster Statistics:")
    print(cluster_stats)
    
    # Cluster sizes
    cluster_sizes = df['KMeans_Cluster'].value_counts().sort_index()
    print(f"\n📊 Cluster Sizes:")
    print(cluster_sizes)
    
    # Silhouette score (use X_scaled and labels, not cluster centers)
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    print(f"\n📏 Overall Silhouette Score: {silhouette_avg:.3f}")
    
    # Create visualizations
    print("\n📊 Generating cluster visualizations...")
    
    # Plot 1: Cluster distribution (pie chart)
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    wedges, texts, autotexts = ax.pie(
        cluster_sizes.values,
        labels=[f'Cluster {i}' for i in cluster_sizes.index],
        autopct='%1.1f%%',
        colors=colors[:len(cluster_sizes)],
        explode=[0.05] * len(cluster_sizes),
        shadow=True
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    ax.set_title('Cluster Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: cluster_distribution.png")
    plt.close()
    
    # Plot 2: Cluster characteristics (radar chart style - parallel coordinates)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Normalize data for comparison - only numeric features
    numeric_features = ['Age', 'Budget (NPR)', 'Duration (days)', 'Spending Score']
    cluster_means = df.groupby('KMeans_Cluster')[numeric_features].mean()
    
    # Plot parallel coordinates
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
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
    plt.savefig('cluster_characteristics.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: cluster_characteristics.png")
    plt.close()
    
    # Plot 3: 2D PCA visualization
    pca = PCA(n_components=2)
    # Use only numeric features for PCA
    numeric_features_for_pca = ['Age', 'Budget (NPR)', 'Duration (days)', 'Spending Score']
    X_pca = pca.fit_transform(df[numeric_features_for_pca])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['KMeans_Cluster'], 
                         cmap='tab10', alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # Add cluster centers (use numeric features only)
    cluster_centers_numeric = kmeans.cluster_centers_[:, :4]  # First 4 features are numeric
    centers_pca = pca.transform(cluster_centers_numeric)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, 
               marker='X', edgecolors='black', linewidth=2, label='Cluster Centers')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('K-Means Clusters (PCA 2D Projection)', fontsize=16, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('clusters_2d_pca.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: clusters_2d_pca.png")
    plt.close()
    
    # Plot 4: Box plots for each feature by cluster
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    features_to_plot = ['Age', 'Budget (NPR)', 'Duration (days)', 'Spending Score']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx // 2, idx % 2]
        df.boxplot(column=feature, by='KMeans_Cluster', ax=ax, patch_artist=True)
        ax.set_title(f'{feature} by Cluster', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel(feature, fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distribution by Cluster', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cluster_boxplots.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: cluster_boxplots.png")
    plt.close()
    
    return cluster_stats, cluster_sizes


def save_model_and_artifacts(kmeans, scaler, encoder, feature_columns, cluster_stats):
    """
    Save trained model and artifacts.
    """
    print("\n💾 Saving model and artifacts...")
    
    # Save K-Means model
    joblib.dump(kmeans, 'kmeans_model.pkl')
    print("✅ Saved: kmeans_model.pkl")
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Saved: scaler.pkl")
    
    # Save encoder
    joblib.dump(encoder, 'activity_encoder.pkl')
    print("✅ Saved: activity_encoder.pkl")
    
    # Save metadata
    metadata = {
        'feature_columns': feature_columns,
        'n_clusters': kmeans.n_clusters,
        'cluster_stats': cluster_stats.to_dict()
    }
    joblib.dump(metadata, 'model_metadata.pkl')
    print("✅ Saved: model_metadata.pkl")


def main():
    print("=" * 70)
    print("🏔️  Tourism Clustering - K-Means Analysis")
    print("=" * 70)
    
    # Load and preprocess data
    X_scaled, df, scaler, encoder, feature_columns, original_clusters = \
        load_and_preprocess_data(r"data\tourism_pokhara.csv")
    
    # Find optimal K
    optimal_k = find_optimal_k(X_scaled, max_k=10)
    
    # Use 5 clusters as specified (or optimal if you prefer)
    n_clusters = 5  # Can change to optimal_k if desired
    print(f"\n🎯 Using {n_clusters} clusters for final model...")
    
    # Apply K-Means
    kmeans, df = apply_kmeans(X_scaled, df, n_clusters=n_clusters)
    
    # Analyze clusters
    cluster_stats, cluster_sizes = analyze_clusters(X_scaled, df, kmeans, feature_columns)
    
    # Save model and artifacts
    save_model_and_artifacts(kmeans, scaler, encoder, feature_columns, cluster_stats)
    
    # Save clustered data
    df.to_csv(r"data\tourism_pokhara_clustered.csv", index=False)
    print("\n💾 Saved: data/tourism_pokhara_clustered.csv")
    
    print("\n" + "=" * 70)
    print("🎉 Clustering Complete!")
    print("=" * 70)
    print(f"\n📊 Number of clusters: {n_clusters}")
    print(f"📏 Silhouette Score: {silhouette_score(X_scaled, kmeans.labels_):.3f}")
    print("\n📁 Generated files:")
    print("   - kmeans_model.pkl")
    print("   - scaler.pkl")
    print("   - activity_encoder.pkl")
    print("   - model_metadata.pkl")
    print("   - optimal_k_analysis.png")
    print("   - cluster_distribution.png")
    print("   - cluster_characteristics.png")
    print("   - clusters_2d_pca.png")
    print("   - cluster_boxplots.png")
    print("\n🚀 Next step: Run 'python tourism_automl_analysis.py' for AutoML!")


if __name__ == "__main__":
    main()
