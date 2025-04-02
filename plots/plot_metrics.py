import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json

def load_metrics_files(metrics_dir):
    """Load and combine all metric CSV files from directory"""
    # Get the run name from the metrics directory
    run_name = os.path.basename(metrics_dir)
    
    all_files = glob.glob(os.path.join(metrics_dir, "*.csv"))
    
    # Dictionary to store metric dataframes
    metrics_data = {}
    test_metrics = {}
    
    for file_path in all_files:
        metric_name = os.path.basename(file_path).replace('.csv', '')
        df = pd.read_csv(file_path)
        
        # Handle MLflow-style CSV files
        if 'Run' in df.columns:
            if 'metric' in df.columns:
                # This is a training history file
                metrics = df.pivot(index='step', columns='metric', values='value')
                for col in metrics.columns:
                    metrics_data[col] = pd.DataFrame(metrics[col])
            else:
                # This is a test metrics file
                if metric_name.startswith('test_'):
                    test_metrics[metric_name] = df.iloc[-1][metric_name]
            continue
            
        # Handle regular training history files
        if 'step' in df.columns:
            if len(df) > 1:  # Multiple steps means it's a history
                metrics_data[metric_name] = df[['step', 'value']].rename(
                    columns={'value': metric_name}).set_index('step')
            elif metric_name.startswith('test_'):
                # Single-value test metric
                test_metrics[metric_name] = df['value'].iloc[0]
    
    # Combine all history metrics into one dataframe
    if metrics_data:
        history_df = pd.concat(metrics_data.values(), axis=1)
        history_df.index.name = 'epoch'
        history_df = history_df.reset_index()
    else:
        history_df = pd.DataFrame()
    
    return history_df, test_metrics, run_name

def plot_metrics(metrics_dir, output_dir='./reports'):
    """Plot metrics from CSV files in the metrics directory"""
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette(['#1f77b4', '#ff7f0e', '#d62728'])  # Blue, Orange, Red
    
    # Load all metrics
    print(f"Loading metrics from {metrics_dir}...")
    history_df, test_metrics, run_name = load_metrics_files(metrics_dir)
    
    # Create run-specific output directory
    run_output_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    if history_df.empty:
        print("No training history data found. Please check your CSV files.")
        return
    
    print(f"Found metrics: {', '.join(history_df.columns[1:])}")
    print(f"Found test metrics: {', '.join(test_metrics.keys())}")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Model Training Metrics - Run: {run_name}', fontsize=16, y=1.05)
    
    # Identify the loss and accuracy columns
    loss_cols = [col for col in history_df.columns if 'loss' in col.lower()]
    acc_cols = [col for col in history_df.columns if 'acc' in col.lower() or 'accuracy' in col.lower()]
    
    # Get final values for legend
    for col in loss_cols:
        if col in history_df.columns:
            ax1.plot(history_df['epoch'], history_df[col], 
                    label=f'{col}: {history_df[col].iloc[-1]:.4f}', 
                    linewidth=2)
    
    # Add test loss horizontal line if available
    if 'test_loss' in test_metrics:
        test_loss = test_metrics['test_loss']
        ax1.axhline(y=test_loss, color='#d62728', linestyle='--', 
                    label=f'Test Loss: {test_loss:.4f}')
    
    ax1.set_title('Loss Over Time', fontsize=14, pad=20)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy
    for col in acc_cols:
        if col in history_df.columns:
            ax2.plot(history_df['epoch'], history_df[col], 
                    label=f'{col}: {history_df[col].iloc[-1]:.4f}', 
                    linewidth=2)
    
    # Add test accuracy horizontal line if available
    if 'test_accuracy' in test_metrics:
        test_accuracy = test_metrics['test_accuracy']
        ax2.axhline(y=test_accuracy, color='#d62728', linestyle='--', 
                    label=f'Test Accuracy: {test_accuracy:.4f}')
    
    ax2.set_title('Accuracy Over Time', fontsize=14, pad=20)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Styling
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # Save the figure
    base_filename = f'training_metrics_{run_name}'
    output_path = os.path.join(run_output_dir, 'training_metrics.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    # Also save a PDF version in run-specific directory
    pdf_path = os.path.join(run_output_dir, 'training_metrics.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Saved plot to {pdf_path}")
    

def plot_roc_curve(json_path, output_dir='./reports'):
    """Plot ROC curve from JSON data containing FPR, TPR and thresholds"""
    run_name = os.path.basename(os.path.dirname(json_path))

    run_output_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 10))
    
    # Load ROC data from JSON
    with open(json_path, 'r') as f:
        roc_data = json.load(f)
    
    # Plot ROC curve
    plt.plot(roc_data['fpr'], roc_data['tpr'], 
            color='#d62728',  # Red
            label=f"Test ROC (AUC = {roc_data['auc']:.3f})",
            linewidth=2)
    
    # Add random classifier reference line
    plt.plot([0, 1], [0, 1], 
            color='gray', 
            linestyle=':', 
            alpha=0.5,
            label='Random Classifier')
    
    # Academic-style formatting
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Receiver Operating Characteristic (ROC) Curve\nRun: {run_name}', 
             fontsize=14, pad=20)
    
    # Set axis limits and grid
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.grid(True, alpha=0.3)
    
    # Style the plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tick_params(labelsize=10)
    
    # Add legend
    plt.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    base_filename = f'roc_curve_{run_name}'
    output_path = os.path.join(run_output_dir, 'roc_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    print(f"Saved ROC curve to {output_path}")
    plt.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot metrics from MLflow CSV files')
    parser.add_argument('--metrics-dir', default='./metrics/ 1743491698798740',
                        help='Directory containing metrics CSV files')
    parser.add_argument('--output-dir', default='./reports',
                        help='Directory to save output plots')
    
    args = parser.parse_args()
    
    plot_metrics(args.metrics_dir, args.output_dir)

    # Look for ROC JSON file in metrics directory
    roc_json = os.path.join(args.metrics_dir, 'test_roc.json')
    if os.path.exists(roc_json):
        print(f"Found ROC curve data at {roc_json}")
        plot_roc_curve(roc_json, args.output_dir)
    else:
        print("No ROC curve data (test_roc.json) found in metrics directory")