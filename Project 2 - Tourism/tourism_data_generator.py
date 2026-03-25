"""
Tourism Dataset Generator for Pokhara
Generates synthetic tourism data with realistic patterns and outliers
"""

import pandas as pd
import numpy as np

def generate_tourism_data(n_samples=5000, random_state=42):
    """
    Generate synthetic tourism dataset for Pokhara.
    
    Features:
    - Age: 18-70 years
    - Budget: 500-10000 NPR per day
    - Duration: 1-30 days
    - Activity Preference: Adventure/Cultural/Relaxation/Spiritual
    - Spending Score: 1-100 (calculated)
    
    Returns:
    - DataFrame with 5 features and cluster labels
    """
    np.random.seed(random_state)
    
    # Initialize empty lists
    ages = []
    budgets = []
    durations = []
    activities = []
    spending_scores = []
    
    # Define cluster patterns
    # Each cluster has different characteristics
    
    # Cluster 0: Budget Backpackers (~25%)
    n_backpackers = int(n_samples * 0.25)
    ages.extend(np.random.randint(18, 32, size=n_backpackers))
    budgets.extend(np.random.exponential(scale=800, size=n_backpackers) + 500)
    durations.extend(np.random.exponential(scale=3, size=n_backpackers) + 1)
    activities.extend(np.random.choice(['Adventure', 'Cultural'], size=n_backpackers, p=[0.6, 0.4]))
    
    # Cluster 1: Cultural Explorers (~20%)
    n_cultural = int(n_samples * 0.20)
    ages.extend(np.random.normal(loc=45, scale=10, size=n_cultural))
    budgets.extend(np.random.normal(loc=3500, scale=800, size=n_cultural))
    durations.extend(np.random.normal(loc=7, scale=2, size=n_cultural))
    activities.extend(np.random.choice(['Cultural', 'Relaxation'], size=n_cultural, p=[0.7, 0.3]))
    
    # Cluster 2: Luxury Travelers (~15%)
    n_luxury = int(n_samples * 0.15)
    ages.extend(np.random.normal(loc=55, scale=10, size=n_luxury))
    budgets.extend(np.random.normal(loc=7500, scale=1500, size=n_luxury))
    durations.extend(np.random.normal(loc=15, scale=5, size=n_luxury))
    activities.extend(np.random.choice(['Relaxation', 'Cultural'], size=n_luxury, p=[0.6, 0.4]))
    
    # Cluster 3: Adventure Seekers (~25%)
    n_adventure = int(n_samples * 0.25)
    ages.extend(np.random.normal(loc=32, scale=8, size=n_adventure))
    budgets.extend(np.random.normal(loc=5000, scale=1200, size=n_adventure))
    durations.extend(np.random.normal(loc=10, scale=3, size=n_adventure))
    activities.extend(np.random.choice(['Adventure', 'Relaxation'], size=n_adventure, p=[0.8, 0.2]))
    
    # Cluster 4: Spiritual Tourists (~15%)
    n_spiritual = int(n_samples * 0.15)
    spiritual_ages = np.concatenate([
        np.random.normal(loc=25, scale=5, size=n_spiritual//3),  # Young spiritual
        np.random.normal(loc=45, scale=10, size=n_spiritual//3),  # Middle-aged
        np.random.normal(loc=60, scale=7, size=n_spiritual - n_spiritual//3 - n_spiritual//3)  # Elderly
    ])
    ages.extend(spiritual_ages)
    
    budgets.extend(np.random.normal(loc=2500, scale=800, size=n_spiritual))
    
    spiritual_durations = np.concatenate([
        np.random.exponential(scale=5, size=n_spiritual//2),  # Short stay
        np.random.normal(loc=15, scale=5, size=n_spiritual - n_spiritual//2)  # Long retreat
    ])
    durations.extend(spiritual_durations)
    
    activities.extend(['Spiritual'] * n_spiritual)
    
    # Convert to numpy arrays for easier manipulation
    ages = np.array(ages)
    budgets = np.array(budgets)
    durations = np.array(durations)
    activities = np.array(activities)
    
    # ========== ADD EXTREME VALUES / OUTLIERS (~5%) ==========
    n_outliers = int(n_samples * 0.05)
    outlier_indices = np.random.choice(len(ages), size=n_outliers, replace=False)
    
    for idx in outlier_indices:
        outlier_type = np.random.choice(['ultra_budget', 'ultra_luxury', 'long_stay', 'young_rich', 'elderly_backpacker'])
        
        if outlier_type == 'ultra_budget':
            ages[idx] = np.random.randint(18, 25)
            budgets[idx] = np.random.randint(300, 600)  # Extremely low budget
            durations[idx] = np.random.randint(1, 3)
            activities[idx] = 'Adventure'
            
        elif outlier_type == 'ultra_luxury':
            ages[idx] = np.random.randint(40, 65)
            budgets[idx] = np.random.randint(9000, 15000)  # Extremely high budget
            durations[idx] = np.random.randint(20, 45)  # Very long stay
            activities[idx] = np.random.choice(['Relaxation', 'Cultural'])
            
        elif outlier_type == 'long_stay':
            ages[idx] = np.random.randint(25, 45)
            budgets[idx] = np.random.randint(1000, 3000)
            durations[idx] = np.random.randint(30, 90)  # Extended stay (digital nomad)
            activities[idx] = np.random.choice(['Cultural', 'Adventure', 'Relaxation'])
            
        elif outlier_type == 'young_rich':
            ages[idx] = np.random.randint(18, 28)
            budgets[idx] = np.random.randint(8000, 12000)  # Young with high budget
            durations[idx] = np.random.randint(3, 10)
            activities[idx] = np.random.choice(['Adventure', 'Relaxation'])
            
        elif outlier_type == 'elderly_backpacker':
            ages[idx] = np.random.randint(60, 75)
            budgets[idx] = np.random.randint(800, 1500)  # Elderly on budget
            durations[idx] = np.random.randint(5, 15)
            activities[idx] = np.random.choice(['Cultural', 'Spiritual'])
    
    # ========== CALCULATE SPENDING SCORE ==========
    # Spending Score = (Budget / max_budget) * 50 + (Duration / max_duration) * 50
    # Normalized to 1-100 scale
    max_budget = 15000
    max_duration = 90
    
    spending_scores = (budgets / max_budget) * 50 + (durations / max_duration) * 50
    spending_scores = np.clip(spending_scores, 1, 100)
    
    # Add some noise to spending score
    spending_scores += np.random.normal(0, 5, size=len(spending_scores))
    spending_scores = np.clip(spending_scores, 1, 100)
    
    # ========== CLIP VALUES TO REALISTIC RANGES ==========
    ages = np.clip(ages, 18, 75)
    budgets = np.clip(budgets, 300, 15000)
    durations = np.clip(durations, 1, 90)
    
    # ========== CREATE DATAFRAME ==========
    df = pd.DataFrame({
        'Age': np.round(ages).astype(int),
        'Budget (NPR)': np.round(budgets, 2),
        'Duration (days)': np.round(durations, 1),
        'Activity Preference': activities,
        'Spending Score': np.round(spending_scores, 1)
    })
    
    # ========== ASSIGN CLUSTER LABELS ==========
    # Based on the original cluster assignment before outliers
    cluster_labels = []
    
    for i in range(len(df)):
        age = df.loc[i, 'Age']
        budget = df.loc[i, 'Budget (NPR)']
        duration = df.loc[i, 'Duration (days)']
        activity = df.loc[i, 'Activity Preference']
        
        # Simple rule-based clustering for ground truth
        if budget < 1500 and age < 35:
            cluster_labels.append(0)  # Budget Backpackers
        elif activity == 'Cultural' and 2500 <= budget <= 5000:
            cluster_labels.append(1)  # Cultural Explorers
        elif budget > 6000 and duration > 10:
            cluster_labels.append(2)  # Luxury Travelers
        elif activity == 'Adventure' and 25 <= age <= 45:
            cluster_labels.append(3)  # Adventure Seekers
        elif activity == 'Spiritual':
            cluster_labels.append(4)  # Spiritual Tourists
        else:
            # Assign to nearest cluster based on characteristics
            if budget < 2000:
                cluster_labels.append(0)
            elif budget > 5000:
                cluster_labels.append(2)
            elif activity == 'Adventure':
                cluster_labels.append(3)
            else:
                cluster_labels.append(1)
    
    df['Cluster'] = cluster_labels
    
    # ========== SHUFFLE DATA ==========
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


def main():
    print("🏔️  Generating Tourism Dataset for Pokhara...")
    print("-" * 60)
    
    # Generate dataset
    df = generate_tourism_data(n_samples=5000)
    
    # Display dataset info
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"\n📋 Columns: {list(df.columns)}")
    print(f"\n📈 Cluster Distribution:")
    print(df['Cluster'].value_counts().sort_index())
    print(f"\n📉 Basic Statistics:")
    print(df.describe())
    print(f"\n🎯 Activity Preference Distribution:")
    print(df['Activity Preference'].value_counts())
    
    # Show extreme values
    print(f"\n🔥 Extreme Values:")
    print(f"   Youngest tourist: {df['Age'].min()} years")
    print(f"   Oldest tourist: {df['Age'].max()} years")
    print(f"   Lowest budget: {df['Budget (NPR)'].min()} NPR")
    print(f"   Highest budget: {df['Budget (NPR)'].max()} NPR")
    print(f"   Shortest stay: {df['Duration (days)'].min()} days")
    print(f"   Longest stay: {df['Duration (days)'].max()} days")
    
    # Save to CSV
    output_path = r"data\tourism_pokhara.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Dataset saved to: {output_path}")
    
    # Show sample
    print(f"\n📄 First 10 rows:")
    print(df.head(10))


if __name__ == "__main__":
    main()
