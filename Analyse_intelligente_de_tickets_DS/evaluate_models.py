"""
Model Evaluation Script
Displays accuracy metrics for all 4 trained models
"""

import json
import os

# Path to metadata
METADATA_PATH = os.path.join(os.path.dirname(__file__), "models", "metadata.json")

def load_metrics():
    """Load model metrics from metadata.json"""
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_separator(char='=', length=80):
    print(char * length)

def display_classification_metrics(model_name, metrics_data):
    """Display metrics for classification models"""
    print(f"\n📊 {model_name.upper()}")
    print_separator('-', 80)
    
    for split_name, split_metrics in metrics_data.items():
        accuracy = split_metrics.get('accuracy', 0) * 100
        f1_macro = split_metrics.get('f1_macro', 0) * 100
        
        print(f"  {split_name.upper():15} | Accuracy: {accuracy:6.2f}% | F1-Score: {f1_macro:6.2f}%")

def display_regression_metrics(model_name, metrics_data):
    """Display metrics for regression models"""
    print(f"\n📊 {model_name.upper()}")
    print_separator('-', 80)
    
    for split_name, split_metrics in metrics_data.items():
        mae = split_metrics.get('mae', 0)
        rmse = split_metrics.get('rmse', 0)
        r2 = split_metrics.get('r2', 0)
        
        print(f"  {split_name.upper():15} | MAE: {mae:6.2f}h | RMSE: {rmse:6.2f}h | R²: {r2:6.4f}")

def main():
    print_separator('=', 80)
    print("🎯 MODEL PERFORMANCE EVALUATION")
    print_separator('=', 80)
    
    metadata = load_metrics()
    
    # 1. Urgency Model (Classification)
    if 'urgency_model' in metadata:
        model_info = metadata['urgency_model']
        print(f"\n🏷️  MODEL 1: URGENCY CLASSIFICATION")
        print(f"    Algorithm: {model_info.get('algorithm', 'N/A')}")
        print(f"    Classes: {', '.join(model_info.get('classes', []))}")
        print(f"    Training samples: {model_info.get('n_train_samples', 'N/A')}")
        display_classification_metrics("Urgency", model_info.get('metrics', {}))
    
    # 2. Category Model (Classification)
    if 'category_model' in metadata:
        model_info = metadata['category_model']
        print(f"\n\n📂 MODEL 2: CATEGORY CLASSIFICATION")
        print(f"    Algorithm: {model_info.get('algorithm', 'N/A')}")
        print(f"    Number of classes: {model_info.get('n_classes', 'N/A')}")
        print(f"    Training samples: {model_info.get('n_train_samples', 'N/A')}")
        display_classification_metrics("Category", model_info.get('metrics', {}))
    
    # 3. Type Model (Classification)
    if 'type_model' in metadata:
        model_info = metadata['type_model']
        print(f"\n\n📝 MODEL 3: TYPE CLASSIFICATION")
        print(f"    Algorithm: {model_info.get('algorithm', 'N/A')}")
        print(f"    Classes: {', '.join(model_info.get('classes', []))}")
        print(f"    Training samples: {model_info.get('n_train_samples', 'N/A')}")
        display_classification_metrics("Type", model_info.get('metrics', {}))
    
    # 4. Time Model (Regression)
    if 'time_model' in metadata:
        model_info = metadata['time_model']
        print(f"\n\n⏱️  MODEL 4: TIME REGRESSION")
        print(f"    Algorithm: {model_info.get('algorithm', 'N/A')}")
        print(f"    Target: {model_info.get('target', 'N/A')}")
        print(f"    Training samples: {model_info.get('n_train_samples', 'N/A')}")
        display_regression_metrics("Resolution Time", model_info.get('metrics', {}))
        
        # Show overfitting analysis if available
        if 'overfitting_analysis' in model_info:
            ov = model_info['overfitting_analysis']
            print(f"\n    📉 Overfitting Analysis:")
            print(f"       Train-Test R² gap: {ov.get('new_train_test_r2_gap', 0):.4f}")
            print(f"       Gap reduction: {ov.get('gap_reduction_percent', 'N/A')}")
    
    print("\n")
    print_separator('=', 80)
    print("✅ Evaluation Complete")
    print_separator('=', 80)

if __name__ == "__main__":
    main()
