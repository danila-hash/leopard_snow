import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
from joblib import Parallel, delayed
from functools import lru_cache

# Read the data with optimized settings
cryptonews = pd.read_csv('popytka2/cryptonews.csv', usecols=['date', 'sentiment'])
btc_price = pd.read_csv('popytka2/btcusd_1-min_data.csv', usecols=['Timestamp', 'Close'])

# Convert and align timestamps, handling null values
cryptonews['date'] = pd.to_datetime(cryptonews['date'], format='mixed', errors='coerce').dt.floor('min')
btc_price['Timestamp'] = pd.to_datetime(btc_price['Timestamp'], unit='s', errors='coerce').dt.floor('min')

# Create a lookup dictionary for faster price access
btc_price_dict = btc_price.set_index('Timestamp')['Close'].to_dict()

# Extract polarity and subjectivity from sentiment dictionary
@lru_cache(maxsize=10000)  # Cache sentiment extraction results
def extract_sentiment(sentiment_str):
    try:
        sentiment_dict = ast.literal_eval(sentiment_str)
        return float(sentiment_dict['polarity']), float(sentiment_dict['subjectivity'])
    except:
        return np.nan, np.nan

# Convert sentiment column to polarity and subjectivity values
cryptonews[['polarity', 'subjectivity']] = pd.DataFrame(
    cryptonews['sentiment'].apply(extract_sentiment).tolist(), 
    index=cryptonews.index
)

# Remove rows with null timestamps or sentiment values
cryptonews = cryptonews.dropna(subset=['date', 'polarity', 'subjectivity'])
btc_price = btc_price.dropna(subset=['Timestamp'])

# Merge news with price data
merged_df = pd.merge_asof(
    cryptonews.sort_values("date"),
    btc_price.sort_values("Timestamp"),
    left_on="date",
    right_on="Timestamp",
    direction="backward"
)

# Optimized function to calculate future price
def calculate_future_price(date, hours):
    future_date = date + pd.Timedelta(hours=hours)
    return btc_price_dict.get(future_date, np.nan)

# Calculate future price changes for different time periods using parallel processing
time_periods = [1, 4, 24]
for hours in time_periods:
    print(f"Calculating {hours}h future prices...")
    # Use vectorized operations instead of parallel processing for this part
    merged_df[f'future_price_{hours}h'] = merged_df['date'].apply(
        lambda x: calculate_future_price(x, hours)
    )
    merged_df[f'price_change_pct_{hours}h'] = ((merged_df[f'future_price_{hours}h'] - merged_df['Close']) / merged_df['Close']) * 100

# Remove rows with null values in price changes
merged_df = merged_df.dropna(subset=[f'price_change_pct_{hours}h' for hours in time_periods])

# Create separate dataframes for positive and negative subjective news
positive_subj_df = merged_df[merged_df['polarity'] > 0].copy()
negative_subj_df = merged_df[merged_df['polarity'] < 0].copy()

# Calculate correlations without parallel processing
print("Calculating correlations...")
correlations = []
for hours in time_periods:
    pearson_corr, _ = pearsonr(merged_df['polarity'], merged_df[f'price_change_pct_{hours}h'])
    spearman_corr, _ = spearmanr(merged_df['polarity'], merged_df[f'price_change_pct_{hours}h'])
    correlations.append((pearson_corr, spearman_corr))

# Calculate subjectivity correlations for both positive and negative news
subjectivity_correlations = []
for hours in time_periods:
    # Positive news correlations
    pos_pearson, _ = pearsonr(positive_subj_df['subjectivity'], positive_subj_df[f'price_change_pct_{hours}h'])
    pos_spearman, _ = spearmanr(positive_subj_df['subjectivity'], positive_subj_df[f'price_change_pct_{hours}h'])
    
    # Negative news correlations
    neg_pearson, _ = pearsonr(negative_subj_df['subjectivity'], negative_subj_df[f'price_change_pct_{hours}h'])
    neg_spearman, _ = spearmanr(negative_subj_df['subjectivity'], negative_subj_df[f'price_change_pct_{hours}h'])
    
    subjectivity_correlations.append((pos_pearson, pos_spearman, neg_pearson, neg_spearman))

# Create subplots for polarity analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('News Polarity vs. BTC Price Changes', fontsize=16, y=1.05)

for idx, (hours, (pearson_corr, spearman_corr)) in enumerate(zip(time_periods, correlations)):
    # Create scatter plot with reduced alpha for better performance
    axes[idx].scatter(merged_df['polarity'], merged_df[f'price_change_pct_{hours}h'], 
                     alpha=0.3, color='blue', label='Price Changes', s=20)
    
    # Add trend line
    z = np.polyfit(merged_df['polarity'], merged_df[f'price_change_pct_{hours}h'], 1)
    p = np.poly1d(z)
    axes[idx].plot(merged_df['polarity'], p(merged_df['polarity']), 
                  color='red', linewidth=2, label='Trend Line')
    
    # Add vertical line at x=0
    axes[idx].axvline(x=0, color='red', linestyle='--', alpha=0.3)
    
    # Customize the plot
    axes[idx].set_title(f'{hours}h Price Change\nPearson: {pearson_corr:.3f}\nSpearman: {spearman_corr:.3f}')
    axes[idx].set_xlabel("News Polarity (Negative to Positive)")
    axes[idx].set_ylabel(f"BTC % Price Change ({hours}h)")
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Create subplots for subjectivity analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('News Subjectivity vs. BTC Price Changes', fontsize=16, y=1.05)

for idx, (hours, (pos_pearson, pos_spearman, neg_pearson, neg_spearman)) in enumerate(zip(time_periods, subjectivity_correlations)):
    # Create scatter plots with reduced alpha for better performance
    axes[idx].scatter(positive_subj_df['subjectivity'], positive_subj_df[f'price_change_pct_{hours}h'], 
                     alpha=0.3, color='green', label='Positive News', s=20)
    axes[idx].scatter(negative_subj_df['subjectivity'], negative_subj_df[f'price_change_pct_{hours}h'], 
                     alpha=0.3, color='red', label='Negative News', s=20)
    
    # Add trend lines
    z_pos = np.polyfit(positive_subj_df['subjectivity'], positive_subj_df[f'price_change_pct_{hours}h'], 1)
    z_neg = np.polyfit(negative_subj_df['subjectivity'], negative_subj_df[f'price_change_pct_{hours}h'], 1)
    p_pos = np.poly1d(z_pos)
    p_neg = np.poly1d(z_neg)
    
    axes[idx].plot(positive_subj_df['subjectivity'], p_pos(positive_subj_df['subjectivity']), 
                  color='green', linewidth=2, label='Positive Trend')
    axes[idx].plot(negative_subj_df['subjectivity'], p_neg(negative_subj_df['subjectivity']), 
                  color='red', linewidth=2, label='Negative Trend')
    
    # Customize the plot
    axes[idx].set_title(f'{hours}h Price Change\nPositive Pearson: {pos_pearson:.3f}\nNegative Pearson: {neg_pearson:.3f}')
    axes[idx].set_xlabel("News Subjectivity")
    axes[idx].set_ylabel(f"BTC % Price Change ({hours}h)")
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend()

# Adjust layout
plt.tight_layout()
plt.show()

# Print correlation statistics
print("\nCorrelation Analysis - Polarity:")
for hours, (pearson_corr, spearman_corr) in zip(time_periods, correlations):
    print(f"\n{hours}h Price Change:")
    print(f"Pearson Correlation: {pearson_corr:.3f}")
    print(f"Spearman Correlation: {spearman_corr:.3f}")

print("\nCorrelation Analysis - Subjectivity:")
for hours, (pos_pearson, pos_spearman, neg_pearson, neg_spearman) in zip(time_periods, subjectivity_correlations):
    print(f"\n{hours}h Price Change:")
    print("Positive News:")
    print(f"Pearson Correlation: {pos_pearson:.3f}")
    print(f"Spearman Correlation: {pos_spearman:.3f}")
    
    print("\nNegative News:")
    print(f"Pearson Correlation: {neg_pearson:.3f}")
    print(f"Spearman Correlation: {neg_spearman:.3f}")

# Granger causality tests with reduced maxlag for faster computation
print("\nGranger Causality Tests - Polarity:")
for hours in time_periods:
    print(f"\n{hours}h Price Change:")
    granger_results = grangercausalitytests(merged_df[['polarity', f'price_change_pct_{hours}h']], 
                                          maxlag=1, verbose=False)  # Reduced from 2 to 1
    print(granger_results)

print("\nGranger Causality Tests - Subjectivity:")
for hours in time_periods:
    print(f"\n{hours}h Price Change:")
    print("Positive News:")
    granger_results = grangercausalitytests(positive_subj_df[['subjectivity', f'price_change_pct_{hours}h']], 
                                          maxlag=1, verbose=False)  # Reduced from 2 to 1
    print(granger_results)
    
    print("\nNegative News:")
    granger_results = grangercausalitytests(negative_subj_df[['subjectivity', f'price_change_pct_{hours}h']], 
                                          maxlag=1, verbose=False)  # Reduced from 2 to 1
    print(granger_results)