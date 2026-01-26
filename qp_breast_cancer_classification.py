import pandas as pd
import numpy as np
import cvxpy as cp
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
    """Prepare data for QP classification with selected features"""
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

def qp_svm_classification(malignant_data, benign_data, C=1.0):
    """
    Quadratic Programming SVM classification
    
    Parameters:
    - malignant_data: numpy array of malignant samples (class +1)
    - benign_data: numpy array of benign samples (class -1)  
    - C: regularization parameter (higher C = less margin, fewer misclassifications)
    """
    print("   🧮 Solving Quadratic Programming (SVM) problem...")
    print(f"   Regularization parameter C: {C}")
    
    # Combine all data
    X = np.vstack((malignant_data, benign_data))
    n_malignant = len(malignant_data)
    n_benign = len(benign_data)
    n_total = n_malignant + n_benign
    
    # Create labels: Malignant = +1, Benign = -1
    y = np.hstack((np.ones(n_malignant), -np.ones(n_benign)))
    
    # Define optimization variables
    w = cp.Variable(2)          # Weight vector [w1, w2]
    gamma = cp.Variable()       # Bias term
    slack = cp.Variable(n_total) # Slack variables
    
    # Constraints
    constraints = []
    
    # SVM constraints: y_i * (w·x_i + gamma) >= 1 - slack_i
    for i in range(n_total):
        constraints.append(y[i] * (X[i] @ w + gamma) >= 1 - slack[i])
    
    # Slack variables must be non-negative
    constraints.append(slack >= 0)
    
    # Objective function: 1/2 * ||w||^2 + C * sum(slack^2)
    # This is the L2-norm soft margin SVM formulation
    objective = cp.Minimize(0.5 * cp.sum_squares(w) + C * cp.sum_squares(slack))
    
    # Create and solve the problem
    prob = cp.Problem(objective, constraints)
    
    # Try different solvers
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        solver_used = "ECOS"
    except:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            solver_used = "SCS"
        except:
            prob.solve(verbose=False)
            solver_used = "Default"
    
    # Check solution status
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        w_opt = w.value
        gamma_opt = gamma.value
        objective_value = prob.value
        
        # Calculate margin
        w_norm = np.linalg.norm(w_opt)
        margin = 2.0 / w_norm if w_norm > 1e-10 else 0.0
        
        print(f"   ✅ QP (SVM) solved successfully using {solver_used}!")
        print(f"   📈 Objective value: {objective_value:.6f}")
        print(f"   🔧 Parameters: w1 = {w_opt[0]:.6f}, w2 = {w_opt[1]:.6f}, gamma = {gamma_opt:.6f}")
        print(f"   📏 Margin width: {margin:.6f}")
        print(f"   📊 ||w||: {w_norm:.6f}")
        
        return w_opt, gamma_opt, margin, objective_value
    else:
        print(f"   ❌ QP (SVM) solution failed with status: {prob.status}")
        return None, None, 0.0, None

def calculate_svm_accuracy(malignant_data, benign_data, w, gamma):
    """Calculate SVM classification accuracy"""
    if w is None or gamma is None:
        return 0.0, (0, 0, 0, 0)
    
    # Malignant predictions (should be positive)
    malignant_scores = malignant_data @ w + gamma
    malignant_predictions = np.sign(malignant_scores)
    malignant_correct = np.sum(malignant_predictions > 0)  # > 0 for malignant
    
    # Benign predictions (should be negative)  
    benign_scores = benign_data @ w + gamma
    benign_predictions = np.sign(benign_scores)
    benign_correct = np.sum(benign_predictions < 0)  # < 0 for benign
    
    total_correct = malignant_correct + benign_correct
    total_samples = len(malignant_data) + len(benign_data)
    accuracy = (total_correct / total_samples) * 100
    
    # Confusion matrix
    true_positives = malignant_correct
    true_negatives = benign_correct
    false_positives = len(benign_data) - benign_correct  # Benign misclassified as Malignant
    false_negatives = len(malignant_data) - malignant_correct  # Malignant misclassified as Benign
    
    return accuracy, (true_positives, true_negatives, false_positives, false_negatives)

def plot_svm_results(malignant, benign, w, gamma, margin, feature1, feature2, accuracy, confusion_matrix, objective_value, C):
    """Plot SVM classification results with maximum margin"""
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
    
    # Plot decision boundary and margins
    if w is not None and abs(w[1]) > 1e-10:
        # Create points for the decision boundary line
        x_plot = np.linspace(x_min - x_padding, x_max + x_padding, 100)
        
        # Decision boundary: w·x + gamma = 0
        y_decision = (-w[0] * x_plot - gamma) / w[1]
        
        # Margin boundaries: w·x + gamma = ±1
        y_margin_upper = (1 - w[0] * x_plot - gamma) / w[1]
        y_margin_lower = (-1 - w[0] * x_plot - gamma) / w[1]
        
        # Plot decision boundary
        valid_decision = (y_decision >= y_min - y_padding) & (y_decision <= y_max + y_padding)
        if np.any(valid_decision):
            ax.plot(x_plot[valid_decision], y_decision[valid_decision], 'purple', 
                   linewidth=3, label='Decision Boundary')
        
        # Plot margin boundaries
        valid_upper = (y_margin_upper >= y_min - y_padding) & (y_margin_upper <= y_max + y_padding)
        valid_lower = (y_margin_lower >= y_min - y_padding) & (y_margin_lower <= y_max + y_padding)
        
        if np.any(valid_upper):
            ax.plot(x_plot[valid_upper], y_margin_upper[valid_upper], 'purple', 
                   linestyle='--', linewidth=1, alpha=0.7, label='Margin')
        if np.any(valid_lower):
            ax.plot(x_plot[valid_lower], y_margin_lower[valid_lower], 'purple', 
                   linestyle='--', linewidth=1, alpha=0.7)
        
        # Fill margin area
        if np.any(valid_upper) and np.any(valid_lower):
            # Find common x-range where both margins are valid
            common_mask = valid_upper & valid_lower
            if np.any(common_mask):
                ax.fill_between(x_plot[common_mask], 
                              y_margin_lower[common_mask], 
                              y_margin_upper[common_mask], 
                              alpha=0.1, color='purple', label='Margin Area')
        
        # Fill class regions
        if np.any(valid_upper):
            ax.fill_between(x_plot[valid_upper], y_margin_upper[valid_upper], 
                          y_max + y_padding, alpha=0.1, color='red', label='Malignant Region')
        if np.any(valid_lower):
            ax.fill_between(x_plot[valid_lower], y_min - y_padding, 
                          y_margin_lower[valid_lower], alpha=0.1, color='blue', label='Benign Region')
    
    # Customize plot with proper labels
    ax.set_xlabel(feature1, fontsize=14)
    ax.set_ylabel(feature2, fontsize=14)
    
    # Extract confusion matrix values
    tp, tn, fp, fn = confusion_matrix
    
    title = (f'Quadratic Programming (SVM) Classification\n'
             f'Accuracy: {accuracy:.1f}% | Margin: {margin:.4f} | C: {C}\n'
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
    plt.savefig(f'images/SVM_C{C}_{feature1_clean}_vs_{feature2_clean}.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def test_svm_classification():
    """Main function to test SVM classification on breast cancer data"""
    print("="*80)
    print("BREAST CANCER CLASSIFICATION - QUADRATIC PROGRAMMING (SVM) TEST")
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
    
    # Test different regularization parameters
    C_values = [0.1, 1.0, 10.0]
    
    results = {}
    
    print("\n" + "="*80)
    print("RUNNING QUADRATIC PROGRAMMING (SVM) CLASSIFICATION")
    print("="*80)
    print(f"Testing {len(valid_feature_pairs)} feature pairs with C values: {C_values}")
    
    # Step 4: Test each feature pair with different C values
    for i, (feature1, feature2) in enumerate(valid_feature_pairs, 1):
        print(f"\n🔬 TEST {i}: {feature1} vs {feature2}")
        print("-" * 60)
        
        try:
            # Prepare data
            malignant, benign = prepare_classification_data(X, y, feature1, feature2)
            
            if malignant is None or benign is None:
                continue
                
            if len(malignant) > 0 and len(benign) > 0:
                feature_results = {}
                
                for C in C_values:
                    print(f"   Testing C = {C}")
                    
                    # Perform SVM classification
                    w_svm, gamma_svm, margin, obj_svm = qp_svm_classification(malignant, benign, C)
                    
                    if w_svm is not None:
                        # Calculate accuracy
                        accuracy, confusion_matrix = calculate_svm_accuracy(malignant, benign, w_svm, gamma_svm)
                        
                        feature_results[C] = {
                            'accuracy': accuracy,
                            'margin': margin,
                            'objective_value': obj_svm,
                            'parameters': (w_svm, gamma_svm),
                            'confusion_matrix': confusion_matrix
                        }
                        
                        print(f"      ✅ Accuracy: {accuracy:.2f}%, Margin: {margin:.4f}")
                        
                        # Plot results for this C value
                        plot_svm_results(malignant, benign, w_svm, gamma_svm, margin, 
                                       feature1, feature2, accuracy, confusion_matrix, obj_svm, C)
                    else:
                        print(f"      ❌ SVM classification failed for C = {C}")
                
                # Store results for this feature pair
                results[(feature1, feature2)] = feature_results
                
            else:
                print("   ⚠️  Skipping - insufficient data for classification")
                
        except Exception as e:
            print(f"   ❌ Error in classification: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 5: Print comprehensive summary
    print("\n" + "="*80)
    print("QUADRATIC PROGRAMMING (SVM) CLASSIFICATION SUMMARY")
    print("="*80)
    
    if results:
        # Print detailed results for each feature pair
        for features, c_results in results.items():
            feature1, feature2 = features
            print(f"\n📊 Feature Pair: {feature1} vs {feature2}")
            print("-" * 80)
            print(f"{'C':<8} {'Accuracy':<10} {'Margin':<12} {'Objective':<12} {'TP/TN/FP/FN':<15}")
            print("-" * 80)
            
            for C, metrics in sorted(c_results.items()):
                tp, tn, fp, fn = metrics['confusion_matrix']
                print(f"{C:<8} {metrics['accuracy']:<9.1f}% {metrics['margin']:<11.4f} "
                      f"{metrics['objective_value']:<11.4f} {tp}/{tn}/{fp}/{fn:<15}")
        
        # Find best performing configuration overall
        best_accuracy = 0
        best_config = None
        
        for features, c_results in results.items():
            for C, metrics in c_results.items():
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    best_config = (features, C, metrics)
        
        if best_config:
            best_features, best_C, best_metrics = best_config
            feature1, feature2 = best_features
            
            print(f"\n🏆 BEST OVERALL PERFORMANCE:")
            print(f"   Feature Pair: {feature1} vs {feature2}")
            print(f"   Regularization C: {best_C}")
            print(f"   Accuracy: {best_metrics['accuracy']:.2f}%")
            print(f"   Margin: {best_metrics['margin']:.6f}")
            print(f"   Objective Value: {best_metrics['objective_value']:.6f}")
            
            # Analyze the effect of C on performance
            print(f"\n📈 Effect of Regularization Parameter C:")
            print("-" * 60)
            for features, c_results in results.items():
                feature1, feature2 = features
                print(f"\n{feature1} vs {feature2}:")
                for C, metrics in sorted(c_results.items()):
                    print(f"  C={C}: Accuracy={metrics['accuracy']:.1f}%, "
                          f"Margin={metrics['margin']:.4f}, ||w||={np.linalg.norm(metrics['parameters'][0]):.4f}")
        
    else:
        print("No successful classifications completed.")
    
    return results

def print_svm_formulation():
    """Print the mathematical formulation of the SVM problem"""
    print("\n" + "="*80)
    print("QUADRATIC PROGRAMMING (SVM) FORMULATION")
    print("="*80)
    print("\nMathematical Formulation (L2-norm Soft Margin SVM):")
    print("Objective: minimize ½‖w‖² + C∑ξᵢ²")
    print("Subject to:")
    print("  yᵢ(w·xᵢ + γ) ≥ 1 - ξᵢ,  for i = 1,...,n")
    print("  ξᵢ ≥ 0")
    print("\nWhere:")
    print("  w = [w₁, w₂] - weight vector")
    print("  γ - bias term")
    print("  xᵢ - feature vectors")
    print("  yᵢ - labels (+1 for Malignant, -1 for Benign)")
    print("  ξᵢ - slack variables for misclassifications")
    print("  C - regularization parameter")
    print("\nClassification Rule:")
    print("  If w·x + γ ≥ 0 → Malignant")
    print("  If w·x + γ < 0 → Benign")
    print("\nMargin Calculation:")
    print("  Margin = 2 / ‖w‖")
    print("\nRegularization Parameter C:")
    print("  Small C: Larger margin, more misclassifications allowed")
    print("  Large C: Smaller margin, fewer misclassifications")
    print("  C → ∞: Hard margin SVM (no misclassifications allowed)")

# Run the SVM classification test
if __name__ == "__main__":
    print_svm_formulation()
    results = test_svm_classification()
    
    print("\n✅ Quadratic Programming (SVM) classification test completed!")
    print("📁 Check the 'images' folder for generated plots (files starting with 'SVM_').")
    print(f"📊 Tested {len(results)} feature pairs successfully.")