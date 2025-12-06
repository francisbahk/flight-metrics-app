import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple

def load_data(filepath: str) -> Tuple[pd.DataFrame, List[str], StandardScaler]:
    """
    Load the CSV file, identify metric columns, and standardize them.
    
    Returns:
        DataFrame (with standardized metrics), list of metric column names, and fitted scaler
    """
    df = pd.read_csv(filepath)
    
    # Exclude non-metric columns
    exclude_cols = {'product_name', 'brand', 'description', 'index'}
    metric_cols = [col for col in df.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"Loaded {len(df)} headphones")
    print(f"Metric columns: {metric_cols}")
    
    # Standardize numeric features (z-score normalization)
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[metric_cols] = scaler.fit_transform(df[metric_cols])
    
    print(f"\nFeature scaling applied (z-score normalization)")
    print(f"Mean of scaled features: ~0, Std: ~1")
    
    return df_scaled, metric_cols, scaler

def ranking_to_dict(ranking: List[int]) -> Dict[int, int]:
    """
    Convert a ranking list (ordered by preference) to a dict mapping index → rank.
    Rank 1 = best (first in list), larger ranks = worse.
    """
    return {idx: rank + 1 for rank, idx in enumerate(ranking)}

def generate_pairwise_samples(
    df: pd.DataFrame,
    ranking: List[int],
    metric_cols: List[str],
    n_samples: int = 10000,
    q: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate pairwise comparison samples from a ranking.
    
    Args:
        df: DataFrame with headphone data (with standardized metrics)
        ranking: List of indices in order of preference
        metric_cols: List of metric column names
        n_samples: Number of pairwise comparisons to generate
        q: Probability of choosing the better-ranked item (default 0.95)
    
    Returns:
        X: (n_samples, n_features) feature differences
        y: (n_samples,) binary labels (1 = better chosen, 0 = worse chosen)
    """
    rank_dict = ranking_to_dict(ranking)
    n_features = len(metric_cols)
    
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
    
    # Set index as DataFrame index for easy lookup
    df_indexed = df.set_index('index')
    
    for i in range(n_samples):
        # Randomly choose two different items from the ranking
        idx1, idx2 = np.random.choice(ranking, size=2, replace=False)
        
        rank1 = rank_dict[idx1]
        rank2 = rank_dict[idx2]
        
        # Determine which is better (lower rank = better)
        if rank1 < rank2:
            better_idx, worse_idx = idx1, idx2
        else:
            better_idx, worse_idx = idx2, idx1
        
        # Get metric values for both items (already standardized)
        better_values = df_indexed.loc[better_idx, metric_cols].values
        worse_values = df_indexed.loc[worse_idx, metric_cols].values
        
        # Feature vector = better - worse
        X[i, :] = better_values - worse_values
        
        # With probability q, human prefers better; else prefers worse
        if np.random.rand() < q:
            y[i] = 1  # Better item chosen
        else:
            y[i] = 0  # Worse item chosen
    
    return X, y

def train_logistic_model(
    X: np.ndarray,
    y: np.ndarray,
    metric_cols: List[str]
) -> Tuple[float, Dict[str, float]]:
    """
    Train logistic regression and extract weights.
    
    Returns:
        intercept (alpha) and dict of {metric: weight}
    """
    model = LogisticRegression(max_iter=5000)
    model.fit(X, y)
    
    intercept = model.intercept_[0]
    weights = {metric: coef for metric, coef in zip(metric_cols, model.coef_[0])}
    
    return intercept, weights

def print_weights_yaml(weights: Dict[str, float], model_name: str, human_sol: List[int]):
    """Print weights in YAML format for easy copying."""
    print(f"\n{model_name}:")
    print(f"  weights:")
    
    # Standard order of metrics (adjust based on your actual columns)
    metric_order = [
        'product_name', 'brand', 'price', 'type', 'connectivity',
        'noise_cancellation', 'battery_life', 'bluetooth_version',
        'driver_size', 'weight', 'water_resistance', 'microphone',
        'review_rating', 'review_count', 'description'
    ]
    
    for metric in metric_order:
        if metric in ['product_name', 'brand', 'description']:
            print(f"    {metric}: 0")
        elif metric in weights:
            print(f"    {metric}: {weights[metric]:.6f}")
        else:
            print(f"    {metric}: 0")
    
    print(f"  human_sol: {human_sol}")

def main():
    # Human rankings
    price_ranking = [55, 33, 44, 43, 75, 25, 58, 36, 26, 35]
    review_ranking = [10, 29, 22, 4, 9, 69, 11, 65, 26, 32]
    
    # Load data and standardize features
    df_scaled, metric_cols, scaler = load_data('input/final_headphones_with_index.csv')
    
    print("\n" + "="*60)
    print("TRAINING PRICE MODEL")
    print("="*60)
    
    # Generate pairwise samples for price ranking (q=0.95)
    X_price, y_price = generate_pairwise_samples(
        df_scaled, price_ranking, metric_cols, n_samples=10000, q=0.95
    )
    print(f"Generated {len(X_price)} pairwise samples")
    print(f"Better item chosen: {np.sum(y_price)} times ({100*np.mean(y_price):.1f}%)")
    
    # Train price model
    price_intercept, price_weights = train_logistic_model(X_price, y_price, metric_cols)
    
    print(f"\nPRICE MODEL RESULTS:")
    print(f"Intercept (alpha): {price_intercept:.6f}")
    print(f"\nLearned Metric Weights (beta):")
    for metric, weight in sorted(price_weights.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {metric:20s}: {weight:+.6f}")
    
    print("\n" + "="*60)
    print("TRAINING REVIEW MODEL")
    print("="*60)
    
    # Generate pairwise samples for review ranking (q=0.95)
    X_review, y_review = generate_pairwise_samples(
        df_scaled, review_ranking, metric_cols, n_samples=10000, q=0.95
    )
    print(f"Generated {len(X_review)} pairwise samples")
    print(f"Better item chosen: {np.sum(y_review)} times ({100*np.mean(y_review):.1f}%)")
    
    # Train review model
    review_intercept, review_weights = train_logistic_model(X_review, y_review, metric_cols)
    
    print(f"\nREVIEW MODEL RESULTS:")
    print(f"Intercept (alpha): {review_intercept:.6f}")
    print(f"\nLearned Metric Weights (beta):")
    for metric, weight in sorted(review_weights.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {metric:20s}: {weight:+.6f}")
    
    print("\n" + "="*60)
    print("YAML FORMAT OUTPUT")
    print("="*60)
    
    # Print PRICE weights in YAML
    print_weights_yaml(price_weights, "PRICE", price_ranking)
    
    # Print REVIEW weights in YAML
    print_weights_yaml(review_weights, "REVIEW", review_ranking)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n✓ All features were standardized (z-score) before training")
    print("✓ These are the LEARNED weights directly from human rankings")
    print("✓ Positive weights → higher values increase item utility")
    print("✓ Negative weights → higher values decrease item utility")
    print("\n✓ Use these weights as-is in your configuration")
    print("✓ They capture the implicit preferences from the human rankings")
    print("\n⚠ Note: Weights are on standardized scale")
    print("⚠ Small sample size (10 items) means weights may have some noise")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()