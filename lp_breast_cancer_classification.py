import pandas as pd
import numpy as np
import pulp as lp
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import os

# Create output directory
os.makedirs('images', exist_ok=True)

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

def load_breast_cancer_data():
    """Load breast cancer dataset from UCI repository"""
    print("Loading Breast Cancer Wisconsin Diagnostic Dataset...")
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
    
    # Data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets
    
    # Convert target to binary (M = 0, B = 1)
    y_binary = (y['Diagnosis'] == 'B').astype(int).values  # 1 for Benign, 0 for Malignant
    
    print("✅ Dataset loaded successfully!")
    print(f"📊 Features shape: {X.shape}")
    print(f"🎯 Target distribution:")
    print(f"   Malignant (M): {sum(y_binary == 0)} samples")
    print(f"   Benign (B): {sum(y_binary == 1)} samples")
    
    return X, y_binary

def analyze_feature_separability(X, y):
    """Analyze which features best separate malignant vs benign tumors"""
    print("\n" + "="*60)
    print("ANALYZING FEATURE SEPARABILITY")
    print("="*60)
    
    malignant_mask = (y == 0)
    benign_mask = (y == 1)
    
    feature_scores = {}
    
    # Analyze ALL features
    for feature in X.columns:
        mal_mean = X[malignant_mask][feature].mean()
        ben_mean = X[benign_mask][feature].mean()
        mal_std = X[malignant_mask][feature].std()
        ben_std = X[benign_mask][feature].std()
        
        # Separability score (higher is better)
        mean_diff = abs(mal_mean - ben_mean)
        pooled_std = np.sqrt(mal_std**2 + ben_std**2)
        separability = mean_diff / pooled_std if pooled_std > 0 else 0
        
        feature_scores[feature] = {
            'separability': separability,
            'malignant_mean': mal_mean,
            'benign_mean': ben_mean,
            'mean_difference': mean_diff
        }
    
    # Sort features by separability score
    ranked_features = sorted(feature_scores.items(), key=lambda x: x[1]['separability'], reverse=True)
    
    print("\n🏆 Top 10 Most Discriminative Features:")
    print("-" * 100)
    print(f"{'Rank':<4} {'Feature':<25} {'Separability':<12} {'M Mean':<10} {'B Mean':<10} {'Diff':<10}")
    print("-" * 100)
    for i, (feature, scores) in enumerate(ranked_features[:10], 1):
        print(f"{i:<4} {feature:<25} {scores['separability']:<12.4f} "
              f"{scores['malignant_mean']:<10.2f} {scores['benign_mean']:<10.2f} "
              f"{scores['mean_difference']:<10.2f}")
    
    return [feature for feature, _ in ranked_features[:10]]

def prepare_classification_data(X, y, feature1, feature2):
    """Prepare data for LP classification with selected features"""
    print(f"\n📋 Preparing data for: {feature1} vs {feature2}")
    
    # Check if features exist in the dataset
    if feature1 not in X.columns:
        print(f"   ❌ Feature '{feature1}' not found in dataset!")
        return None, None
    if feature2 not in X.columns:
        print(f"   ❌ Feature '{feature2}' not found in dataset!")
        return None, None
    
    malignant = X[y == 0][[feature1, feature2]].values
    benign = X[y == 1][[feature1, feature2]].values
    
    print(f"   Malignant samples: {len(malignant)}")
    print(f"   Benign samples: {len(benign)}")
    print(f"   Total samples: {len(malignant) + len(benign)}")
    
    # Print basic statistics
    print(f"   {feature1} - Malignant: {malignant[:, 0].mean():.2f} ± {malignant[:, 0].std():.2f}")
    print(f"   {feature1} - Benign: {benign[:, 0].mean():.2f} ± {benign[:, 0].std():.2f}")
    print(f"   {feature2} - Malignant: {malignant[:, 1].mean():.2f} ± {malignant[:, 1].std():.2f}")
    print(f"   {feature2} - Benign: {benign[:, 1].mean():.2f} ± {benign[:, 1].std():.2f}")
    
    return malignant, benign

def lp_classification_fixed(malignant_data, benign_data):
    """FIXED Linear Programming classification - Proper formulation"""
    print("   🧮 Solving Linear Programming problem...")
    
    # Combine all data
    X = np.vstack((malignant_data, benign_data))
    n_malignant = len(malignant_data)
    n_benign = len(benign_data)
    n_total = n_malignant + n_benign
    
    # Create LP problem
    prob = lp.LpProblem("BreastCancer_Classification_Fixed", lp.LpMinimize)
    
    # Decision variables for the hyperplane: w1*x1 + w2*x2 + gamma = 0
    w1 = lp.LpVariable("w1", cat="Continuous")
    w2 = lp.LpVariable("w2", cat="Continuous")
    gamma = lp.LpVariable("gamma", cat="Continuous")
    
    # Slack variables for each data point
    slack = [lp.LpVariable(f"slack_{i}", lowBound=0) for i in range(n_total)]
    
    # Objective function: minimize total slack (misclassification cost)
    prob += lp.lpSum(slack), "Total_Misclassification_Cost"
    
    # Constraints for MALIGNANT points (class +1): w·x + gamma >= 1 - slack
    for i in range(n_malignant):
        prob += (w1 * malignant_data[i, 0] + w2 * malignant_data[i, 1] + gamma >= 1 - slack[i]), f"Malignant_{i}"
    
    # Constraints for BENIGN points (class -1): w·x + gamma <= -1 + slack
    for i in range(n_benign):
        j = i + n_malignant  # index in combined data
        prob += (w1 * benign_data[i, 0] + w2 * benign_data[i, 1] + gamma <= -1 + slack[j]), f"Benign_{i}"
    
    # Solve the problem
    solver = lp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    
    # Extract results
    if prob.status == lp.LpStatusOptimal:
        w_opt = np.array([w1.varValue, w2.varValue])
        gamma_opt = gamma.varValue
        objective_value = prob.objective.value()
        
        print(f"   ✅ LP solved successfully!")
        print(f"   📈 Objective value (total slack): {objective_value:.6f}")
        print(f"   🔧 Parameters: w1 = {w_opt[0]:.6f}, w2 = {w_opt[1]:.6f}, gamma = {gamma_opt:.6f}")
        
        return w_opt, gamma_opt, objective_value
    else:
        print(f"   ❌ LP solution failed with status: {prob.status}")
        return None, None, None

def calculate_accuracy_fixed(malignant_data, benign_data, w, gamma):
    """Calculate classification accuracy with proper class assignment"""
    if w is None or gamma is None:
        return 0.0, (0, 0, 0, 0)
    
    # Malignant should be classified as +1 (w·x + gamma >= 0)
    malignant_predictions = np.sign(malignant_data @ w + gamma)
    malignant_correct = np.sum(malignant_predictions >= 0)  # Should be positive
    
    # Benign should be classified as -1 (w·x + gamma < 0)
    benign_predictions = np.sign(benign_data @ w + gamma)
    benign_correct = np.sum(benign_predictions < 0)  # Should be negative
    
    total_correct = malignant_correct + benign_correct
    total_samples = len(malignant_data) + len(benign_data)
    accuracy = (total_correct / total_samples) * 100
    
    # Confusion matrix
    true_positives = malignant_correct
    true_negatives = benign_correct
    false_positives = len(benign_data) - benign_correct  # Benign misclassified as Malignant
    false_negatives = len(malignant_data) - malignant_correct  # Malignant misclassified as Benign
    
    return accuracy, (true_positives, true_negatives, false_positives, false_negatives)

def plot_lp_results_fixed(malignant, benign, w, gamma, feature1, feature2, accuracy, confusion_matrix, objective_value):
    """Plot LP classification results with proper separation"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot data points
    ax.scatter(malignant[:, 0], malignant[:, 1], c='red', marker='o', 
               label='Malignant (M)', s=60, alpha=0.8, edgecolors='black', linewidth=0.8)
    ax.scatter(benign[:, 0], benign[:, 1], c='blue', marker='s', 
               label='Benign (B)', s=60, alpha=0.8, edgecolors='black', linewidth=0.8)
    
    # Set proper axis limits with some padding
    x_min = min(malignant[:, 0].min(), benign[:, 0].min())
    x_max = max(malignant[:, 0].max(), benign[:, 0].max())
    y_min = min(malignant[:, 1].min(), benign[:, 1].min())
    y_max = max(malignant[:, 1].max(), benign[:, 1].max())
    
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Plot decision boundary (w·x + gamma = 0)
    if w is not None and abs(w[1]) > 1e-10:
        # Create points for the decision boundary line
        x_plot = np.linspace(x_min - x_padding, x_max + x_padding, 100)
        y_plot = (-w[0] * x_plot - gamma) / w[1]
        
        # Only plot within the data range
        valid_mask = (y_plot >= y_min - y_padding) & (y_plot <= y_max + y_padding)
        if np.any(valid_mask):
            ax.plot(x_plot[valid_mask], y_plot[valid_mask], 'green', linewidth=3, label='Decision Boundary')
            
            # Plot margin boundaries (w·x + gamma = ±1)
            y_margin_upper = (1 - w[0] * x_plot - gamma) / w[1]
            y_margin_lower = (-1 - w[0] * x_plot - gamma) / w[1]
            
            upper_mask = (y_margin_upper >= y_min - y_padding) & (y_margin_upper <= y_max + y_padding)
            lower_mask = (y_margin_lower >= y_min - y_padding) & (y_margin_lower <= y_max + y_padding)
            
            if np.any(upper_mask):
                ax.plot(x_plot[upper_mask], y_margin_upper[upper_mask], 'green', linestyle='--', 
                       linewidth=1, alpha=0.6, label='Margin')
            if np.any(lower_mask):
                ax.plot(x_plot[lower_mask], y_margin_lower[lower_mask], 'green', linestyle='--', 
                       linewidth=1, alpha=0.6)
            
            # Fill the region for malignant class (above upper margin)
            if np.any(upper_mask):
                ax.fill_between(x_plot[upper_mask], y_margin_upper[upper_mask], y_max + y_padding, 
                              alpha=0.1, color='red', label='Malignant Region')
            
            # Fill the region for benign class (below lower margin)
            if np.any(lower_mask):
                ax.fill_between(x_plot[lower_mask], y_min - y_padding, y_margin_lower[lower_mask], 
                              alpha=0.1, color='blue', label='Benign Region')
    
    # Customize plot with proper labels
    ax.set_xlabel(feature1, fontsize=14)
    ax.set_ylabel(feature2, fontsize=14)
    
    # Extract confusion matrix values
    tp, tn, fp, fn = confusion_matrix
    
    title = (f'Linear Programming Classification (Fixed)\n'
             f'Accuracy: {accuracy:.1f}% | Total Slack: {objective_value:.4f}\n'
             f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    ax.set_title(title, fontsize=14, pad=20)
    
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add equation text
    if w is not None:
        equation = f'Decision Boundary: {w[0]:.3f}·{feature1} + {w[1]:.3f}·{feature2} + {gamma:.3f} = 0'
        ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    feature1_clean = feature1.replace(' ', '_').replace('(', '').replace(')', '')
    feature2_clean = feature2.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f'images/LP_FIXED_{feature1_clean}_vs_{feature2_clean}.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def test_lp_classification_fixed():
    """Main function to test FIXED LP classification on breast cancer data"""
    print("="*80)
    print("BREAST CANCER CLASSIFICATION - FIXED LINEAR PROGRAMMING TEST")
    print("="*80)
    
    # Step 1: Load data
    X, y = load_breast_cancer_data()
    
    # Step 2: Analyze feature separability
    top_features = analyze_feature_separability(X, y)
    
    # Step 3: Define feature pairs to test - use the most discriminative features
    feature_pairs = [
        (top_features[0], top_features[1]) if len(top_features) >= 2 else ('concave_points1', 'radius1'),
        ('concave_points1', 'radius1'),    # Highly discriminative features
        ('perimeter1', 'area1'),           # Geometric measurements
        ('texture1', 'smoothness1'),       # Texture characteristics
    ]
    
    # Filter out pairs where features don't exist
    valid_feature_pairs = []
    for feature1, feature2 in feature_pairs:
        if feature1 in X.columns and feature2 in X.columns:
            valid_feature_pairs.append((feature1, feature2))
        else:
            print(f"⚠️  Skipping pair ({feature1}, {feature2}) - features not found")
    
    if not valid_feature_pairs:
        # Use first two available features as fallback
        available_features = list(X.columns[:2])
        valid_feature_pairs = [(available_features[0], available_features[1])]
        print(f"Using fallback features: {available_features[0]} vs {available_features[1]}")
    
    results = {}
    
    print("\n" + "="*80)
    print("RUNNING FIXED LINEAR PROGRAMMING CLASSIFICATION")
    print("="*80)
    print(f"Testing {len(valid_feature_pairs)} feature pairs...")
    
    # Step 4: Test each feature pair with FIXED formulation
    for i, (feature1, feature2) in enumerate(valid_feature_pairs, 1):
        print(f"\n🔬 TEST {i}: {feature1} vs {feature2}")
        print("-" * 60)
        
        try:
            # Prepare data
            malignant, benign = prepare_classification_data(X, y, feature1, feature2)
            
            if malignant is None or benign is None:
                continue
                
            if len(malignant) > 0 and len(benign) > 0:
                # Perform FIXED LP classification
                w_lp, gamma_lp, obj_lp = lp_classification_fixed(malignant, benign)
                
                if w_lp is not None:
                    # Calculate accuracy with FIXED method
                    accuracy, confusion_matrix = calculate_accuracy_fixed(malignant, benign, w_lp, gamma_lp)
                    
                    # Store results
                    results[(feature1, feature2)] = {
                        'accuracy': accuracy,
                        'objective_value': obj_lp,
                        'parameters': (w_lp, gamma_lp),
                        'confusion_matrix': confusion_matrix
                    }
                    
                    print(f"   ✅ Classification Accuracy: {accuracy:.2f}%")
                    
                    # Plot results with FIXED plotting
                    plot_lp_results_fixed(malignant, benign, w_lp, gamma_lp, feature1, feature2, 
                                        accuracy, confusion_matrix, obj_lp)
                else:
                    print("   ❌ LP classification failed")
                
            else:
                print("   ⚠️  Skipping - insufficient data for classification")
                
        except Exception as e:
            print(f"   ❌ Error in classification: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 5: Print summary
    print("\n" + "="*80)
    print("FIXED LINEAR PROGRAMMING CLASSIFICATION SUMMARY")
    print("="*80)
    
    if results:
        summary_data = []
        for features, metrics in results.items():
            feature1, feature2 = features
            tp, tn, fp, fn = metrics['confusion_matrix']
            
            summary_data.append({
                'Feature Pair': f"{feature1} vs {feature2}",
                'Accuracy': f"{metrics['accuracy']:.1f}%",
                'Total Slack': f"{metrics['objective_value']:.4f}",
                'TP/TN/FP/FN': f"{tp}/{tn}/{fp}/{fn}"
            })
        
        # Create pretty summary table
        print("\n" + "Summary of FIXED LP Classification Results:")
        print("-" * 80)
        for item in summary_data:
            print(f"Features: {item['Feature Pair']}")
            print(f"  Accuracy: {item['Accuracy']}")
            print(f"  Total Slack: {item['Total Slack']}")
            print(f"  Confusion Matrix: {item['TP/TN/FP/FN']}")
            print("-" * 80)
        
        # Find best performing pair
        best_pair = max(results.items(), key=lambda x: x[1]['accuracy'])
        best_features, best_metrics = best_pair
        
        print(f"\n🏆 BEST PERFORMING FEATURE PAIR:")
        print(f"   {best_features[0]} vs {best_features[1]}")
        print(f"   Accuracy: {best_metrics['accuracy']:.2f}%")
        print(f"   Total Slack: {best_metrics['objective_value']:.6f}")
        
    else:
        print("No successful classifications completed.")
    
    return results

def print_fixed_lp_formulation():
    """Print the CORRECT mathematical formulation of the LP problem"""
    print("\n" + "="*80)
    print("CORRECT LINEAR PROGRAMMING FORMULATION")
    print("="*80)
    print("\nMathematical Formulation:")
    print("Objective: minimize ∑ξᵢ")
    print("Subject to:")
    print("  w·mᵢ + γ ≥ 1 - ξᵢ,  for i = 1,...,m  (Malignant constraints)")
    print("  w·bⱼ + γ ≤ -1 + ξⱼ, for j = 1,...,k  (Benign constraints)")
    print("  ξᵢ ≥ 0")
    print("\nWhere:")
    print("  w = [w₁, w₂] - weight vector")
    print("  γ - bias term")
    print("  mᵢ - malignant tumor features (class +1)")
    print("  bⱼ - benign tumor features (class -1)")
    print("  ξᵢ - slack variables for misclassifications")
    print("\nClassification Rule:")
    print("  If w·x + γ ≥ 0 → Malignant")
    print("  If w·x + γ < 0 → Benign")

# Run the FIXED LP classification test
if __name__ == "__main__":
    print_fixed_lp_formulation()
    results = test_lp_classification_fixed()
    
    print("\n✅ FIXED Linear Programming classification test completed!")
    print("📁 Check the 'images' folder for generated plots (files starting with 'LP_FIXED_').")
    print(f"📊 Tested {len(results)} feature pairs successfully.")