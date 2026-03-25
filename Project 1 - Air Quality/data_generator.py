"""
Air Quality Dataset Generator for Kathmandu
Generates synthetic pollution monitoring data with realistic patterns
"""

import pandas as pd
import numpy as np

def generate_air_quality_data(n_samples=5000, random_state=42):
    """
    Generate synthetic air quality dataset for Kathmandu.
    
    Features:
    - PM2.5: Fine particulate matter (0-500 µg/m³)
    - PM10: Coarse particulate matter (0-600 µg/m³)
    - Temperature: Ambient temperature (5-40 °C)
    - Humidity: Relative humidity (20-95 %)
    - Wind Speed: Wind velocity (0-30 km/h)
    
    Target:
    - Air Quality Category: Good / Moderate / Unhealthy
    """
    np.random.seed(random_state)
    
    # Generate base features with realistic distributions
    # PM2.5 - skewed towards lower values but with pollution spikes
    pm25 = np.random.exponential(scale=50, size=n_samples)
    pm25 = np.clip(pm25, 0, 500)
    
    # PM10 - correlated with PM2.5 but higher values
    pm10 = pm25 * np.random.uniform(1.2, 2.0, size=n_samples)
    pm10 = np.clip(pm10 + np.random.normal(0, 30, size=n_samples), 0, 600)
    
    # Temperature - seasonal pattern (normal distribution around 20°C)
    temperature = np.random.normal(loc=20, scale=8, size=n_samples)
    temperature = np.clip(temperature, 5, 40)
    
    # Humidity - inverse relationship with temperature
    humidity = 70 - (temperature - 20) * 0.5 + np.random.normal(0, 15, size=n_samples)
    humidity = np.clip(humidity, 20, 95)
    
    # Wind Speed - affects pollution dispersion
    wind_speed = np.random.exponential(scale=5, size=n_samples)
    wind_speed = np.clip(wind_speed, 0, 30)
    
    # Determine Air Quality Category based on PM levels
    # Using WHO and Nepal air quality standards
    air_quality = []
    for p25, p10 in zip(pm25, pm10):
        if p25 < 35 and p10 < 50:
            air_quality.append('Good')
        elif p25 > 150 or p10 > 250:
            air_quality.append('Unhealthy')
        else:
            air_quality.append('Moderate')
    
    # Create DataFrame
    df = pd.DataFrame({
        'PM2.5': np.round(pm25, 2),
        'PM10': np.round(pm10, 2),
        'Temperature': np.round(temperature, 1),
        'Humidity': np.round(humidity, 1),
        'Wind Speed': np.round(wind_speed, 2),
        'Air Quality Category': air_quality
    })
    
    # Add some realistic noise/patterns
    # Winter months have higher pollution (simulate with random assignment)
    winter_indices = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
    df.loc[winter_indices, 'PM2.5'] *= np.random.uniform(1.2, 1.8, size=len(winter_indices))
    df.loc[winter_indices, 'PM10'] *= np.random.uniform(1.2, 1.8, size=len(winter_indices))
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
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


def main():
    print("🌬️  Generating Air Quality Dataset for Kathmandu...")
    print("-" * 50)
    
    # Generate dataset
    df = generate_air_quality_data(n_samples=5000)
    
    # Display dataset info
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"\n📋 Columns: {list(df.columns)}")
    print(f"\n📈 Air Quality Distribution:")
    print(df['Air Quality Category'].value_counts())
    print(f"\n📉 Basic Statistics:")
    print(df.describe())
    
    # Save to CSV
    output_path = r"data\air_quality_kathmandu.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Dataset saved to: {output_path}")
    
    # Show sample
    print(f"\n📄 First 10 rows:")
    print(df.head(10))


if __name__ == "__main__":
    main()
