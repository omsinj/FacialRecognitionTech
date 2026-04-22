"""
analyse_knn_confidence.py

Purpose
-------
Analyse how "confident" a trained KNN classifier is on the TEST split by inspecting:
- how many neighbors voted for the winning class (winning_proportion)
- the margin between the top two vote proportions (decision_margin)
- how these scores differ for correct vs incorrect predictions
- basic statistical separation (t-test + Cohen's d)
- distributions plotted to disk

Outputs
-------
Writes to: META_DIR / evaluation / confidence_analysis /
- knn_detailed_voting_analysis.csv        (per-sample voting details)
- knn_confidence_statistics.json          (summary statistics + separation metrics)
- knn_confidence_distributions.png        (boxplot + histogram)

Assumptions
-----------
- There is a parquet file at META_DIR/embeddings_with_unknown.parquet
- That parquet contains:
    - column "split" with values like "train"/"val"/"test"
    - column "identity" containing ground-truth labels
    - embedding feature columns named as ints (0,1,2,...) OR strings of digits ("0","1",...)
- A trained KNN model exists at MODELS_DIR/classifier.joblib
- A sklearn LabelEncoder exists at MODELS_DIR/label_encoder.joblib
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from scipy import stats
from pathlib import Path

# Get the absolute path to the project root
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import and convert to Path objects
from fr_utils.config import META_DIR, MODELS_DIR

# Convert strings to Path objects
META_DIR = Path(META_DIR)
MODELS_DIR = Path(MODELS_DIR)

def load_test_data():
    """Load test embeddings and labels."""
    print(f"Loading from: {META_DIR / 'embeddings_with_unknown.parquet'}")
    df = pd.read_parquet(META_DIR / "embeddings_with_unknown.parquet")
    
    # Identify embedding feature columns
    feat_cols = [
        c for c in df.columns 
        if isinstance(c, int) or (isinstance(c, str) and c.isdigit())
    ]
    
    # Restrict to test split only
    test_df = df[df["split"] == "test"]
    X_test = test_df[feat_cols].to_numpy(dtype=np.float32)
    y_test = test_df["identity"].to_numpy()
    
    # Debug: Check what labels we have
    print(f"Sample labels: {y_test[:5]}")
    print(f"Label types: {type(y_test[0])}")
    
    return X_test, y_test, feat_cols

def compute_voting_analysis(clf, X_test, y_test, le):
    """
    Compute detailed KNN voting analysis with neighbor inspection.
    """
    if not hasattr(clf, "kneighbors"):
        raise ValueError("Classifier does not support kneighbors method")
    
    n_neighbors = clf.n_neighbors
    print(f"KNN with k={n_neighbors}")
    
    # Get nearest neighbors
    distances, indices = clf.kneighbors(X_test, n_neighbors=n_neighbors)
    
    voting_results = []
    
    for i in range(len(X_test)):
        true_label_raw = y_test[i]
        
        # Convert label to format expected by encoder
        if isinstance(true_label_raw, (int, np.integer)):
            # If label is integer, try different formats
            label_to_encode = str(true_label_raw)
        else:
            label_to_encode = str(true_label_raw)
        
        # Try to encode
        try:
            true_label_enc = le.transform([label_to_encode])[0]
        except ValueError:
            # Try alternative: remove any prefix
            if isinstance(true_label_raw, str) and '_' in true_label_raw:
                # Try getting the number part
                try:
                    num_part = true_label_raw.split('_')[-1]
                    true_label_enc = le.transform([num_part])[0]
                except:
                    # Last resort: use the raw value as integer if possible
                    try:
                        true_label_enc = int(true_label_raw)
                    except:
                        true_label_enc = i  # Fallback
            else:
                try:
                    true_label_enc = int(true_label_raw)
                except:
                    true_label_enc = i
        
        # Get neighbor labels (these are already encoded integers)
        neighbor_labels = clf._y[indices[i]]
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        
        # Voting proportions
        vote_proportions = counts / n_neighbors
        winning_idx = np.argmax(counts)
        winning_label_enc = unique[winning_idx]
        winning_proportion = vote_proportions[winning_idx]
        
        # Get predicted label
        pred_label_enc = clf.predict(X_test[i:i+1])[0]
        
        # Convert back to readable labels
        try:
            true_label_display = le.inverse_transform([true_label_enc])[0]
        except:
            true_label_display = str(true_label_raw)
        
        try:
            pred_label_display = le.inverse_transform([pred_label_enc])[0]
        except:
            pred_label_display = str(pred_label_enc)
        
        # Calculate decision margin (difference between top two)
        if len(vote_proportions) > 1:
            sorted_props = np.sort(vote_proportions)[-2:]
            margin = sorted_props[1] - sorted_props[0]
        else:
            margin = winning_proportion
        
        voting_results.append({
            'sample_id': i,
            'true_label': true_label_display,
            'predicted_label': pred_label_display,
            'true_label_encoded': int(true_label_enc),
            'predicted_label_encoded': int(pred_label_enc),
            'winning_proportion': float(winning_proportion),
            'decision_margin': float(margin),
            'is_correct': bool(pred_label_enc == true_label_enc),
            'num_classes_voted': int(len(unique)),
            'num_neighbors': int(n_neighbors),
            'neighbor_distance_mean': float(np.mean(distances[i])),
            'neighbor_distance_std': float(np.std(distances[i])),
            'neighbor_indices': indices[i].tolist(),
            'neighbor_distances': distances[i].tolist()
        })
    
    return pd.DataFrame(voting_results)

def create_visualizations(df_votes, output_dir):
    """Create confidence distribution visualizations."""
    correct = df_votes[df_votes['is_correct']]
    incorrect = df_votes[~df_votes['is_correct']]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot
    axes[0].boxplot([correct['winning_proportion'], incorrect['winning_proportion']], 
                   labels=['Correct', 'Incorrect'])
    axes[0].set_title('Distribution of KNN Winning Proportions')
    axes[0].set_ylabel('Winning Proportion')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram
    axes[1].hist(correct['winning_proportion'], bins=20, alpha=0.7, label='Correct', density=True)
    axes[1].hist(incorrect['winning_proportion'], bins=20, alpha=0.7, label='Incorrect', density=True)
    axes[1].set_title('Density of Confidence Scores')
    axes[1].set_xlabel('Winning Proportion')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "knn_confidence_distributions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def main():
    """Main execution function."""
    print("=== KNN CONFIDENCE ANALYSIS (with Neighbors) ===\n")
    
    # Create output directory
    output_dir = META_DIR / "evaluation" / "confidence_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("1. Loading test data...")
    X_test, y_test, feat_cols = load_test_data()
    print(f"   Loaded {len(X_test)} test samples")
    
    # Load models
    print("2. Loading models...")
    clf = load(MODELS_DIR / "classifier.joblib")
    le = load(MODELS_DIR / "label_encoder.joblib")
    
    print(f"   Classifier: {type(clf).__name__}")
    print(f"   Classes in encoder: {len(le.classes_)}")
    print(f"   Sample classes: {le.classes_[:5]}")
    
    # Analyze voting
    print("3. Analyzing KNN voting patterns...")
    df_votes = compute_voting_analysis(clf, X_test, y_test, le)
    
    # Save detailed analysis
    csv_path = output_dir / "knn_detailed_voting_analysis.csv"
    df_votes.to_csv(csv_path, index=False)
    print(f"   Detailed analysis saved to {csv_path}")
    
    # Calculate statistics
    print("4. Computing statistics...")
    correct = df_votes[df_votes['is_correct']]
    incorrect = df_votes[~df_votes['is_correct']]
    
    stats_dict = {
        'correct': {
            'count': int(len(correct)),
            'mean': float(correct['winning_proportion'].mean()),
            'std': float(correct['winning_proportion'].std()),
            'median': float(correct['winning_proportion'].median()),
            'min': float(correct['winning_proportion'].min()),
            'max': float(correct['winning_proportion'].max()),
            'ci_95_lower': float(correct['winning_proportion'].mean() - 1.96 * correct['winning_proportion'].std() / np.sqrt(len(correct))),
            'ci_95_upper': float(correct['winning_proportion'].mean() + 1.96 * correct['winning_proportion'].std() / np.sqrt(len(correct)))
        },
        'incorrect': {
            'count': int(len(incorrect)),
            'mean': float(incorrect['winning_proportion'].mean()),
            'std': float(incorrect['winning_proportion'].std()),
            'median': float(incorrect['winning_proportion'].median()),
            'min': float(incorrect['winning_proportion'].min()),
            'max': float(incorrect['winning_proportion'].max())
        }
    }
    
    # Calculate separation metrics
    t_stat, p_value = stats.ttest_ind(correct['winning_proportion'], incorrect['winning_proportion'], equal_var=False)
    cohens_d = (correct['winning_proportion'].mean() - incorrect['winning_proportion'].mean()) / np.sqrt((correct['winning_proportion'].std()**2 + incorrect['winning_proportion'].std()**2) / 2)
    
    stats_dict['separation'] = {
        'mean_difference': float(stats_dict['correct']['mean'] - stats_dict['incorrect']['mean']),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d)
    }
    
    # Save statistics
    stats_path = output_dir / "knn_confidence_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"   Statistics saved to {stats_path}")
    
    # Create visualizations
    print("5. Creating visualizations...")
    plot_path = create_visualizations(df_votes, output_dir)
    print(f"   Visualization saved to {plot_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("KNN CONFIDENCE ANALYSIS RESULTS")
    print("="*60)
    print(f"\nCorrect classifications: {stats_dict['correct']['count']}")
    print(f"Incorrect classifications: {stats_dict['incorrect']['count']}")
    print(f"Accuracy: {stats_dict['correct']['count'] / (stats_dict['correct']['count'] + stats_dict['incorrect']['count']):.3f}")
    print(f"\n--- Confidence Statistics ---")
    print(f"Mean winning proportion (correct):   {stats_dict['correct']['mean']:.3f} ± {stats_dict['correct']['std']:.3f}")
    print(f"Mean winning proportion (incorrect): {stats_dict['incorrect']['mean']:.3f} ± {stats_dict['incorrect']['std']:.3f}")
    print(f"Separation: {stats_dict['separation']['mean_difference']:.3f}")
    print(f"Cohen's d effect size: {stats_dict['separation']['cohens_d']:.3f}")
    print(f"Statistical significance: t = {stats_dict['separation']['t_statistic']:.3f}, p = {stats_dict['separation']['p_value']:.6f}")
    
    # Threshold analysis
    print(f"\n--- Threshold Analysis ---")
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        high_conf_correct = len(correct[correct['winning_proportion'] >= threshold])
        high_conf_incorrect = len(incorrect[incorrect['winning_proportion'] >= threshold])
        
        print(f"Threshold {threshold}: {high_conf_correct}/{len(correct)} ({high_conf_correct/len(correct)*100:.1f}%) correct above")
        print(f"                   {high_conf_incorrect}/{len(incorrect)} ({high_conf_incorrect/len(incorrect)*100:.1f}%) incorrect above")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()