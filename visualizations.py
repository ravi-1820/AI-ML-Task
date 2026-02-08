"""
Visualization utilities for LSTM text generator training.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation loss and accuracy.
    
    Args:
        history: Keras History object from model.fit()
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_architecture_comparison(results, save_path='architecture_comparison.png'):
    """
    Plot comparison of different model architectures.
    
    Args:
        results: Dictionary with architecture names as keys and metrics as values
        save_path: Path to save the plot
    """
    architectures = list(results.keys())
    
    metrics = {
        'Training Loss': [results[arch]['final_loss'] for arch in architectures],
        'Validation Loss': [results[arch]['final_val_loss'] for arch in architectures],
        'Training Accuracy': [results[arch]['final_accuracy'] for arch in architectures],
        'Validation Accuracy': [results[arch]['final_val_accuracy'] for arch in architectures]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        axes[idx].bar(architectures, values, color=colors[idx], alpha=0.7, edgecolor='black')
        axes[idx].set_title(metric_name, fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Value', fontsize=12)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Architecture comparison plot saved to {save_path}")
    plt.close()


def plot_temperature_comparison(generated_texts, temperatures, save_path='temperature_comparison.txt'):
    """
    Save temperature comparison to a text file.
    
    Args:
        generated_texts: List of generated text strings
        temperatures: List of temperature values used
        save_path: Path to save the comparison
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TEMPERATURE COMPARISON FOR TEXT GENERATION\n")
        f.write("=" * 80 + "\n\n")
        
        for temp, text in zip(temperatures, generated_texts):
            f.write(f"\nTemperature: {temp}\n")
            f.write("-" * 80 + "\n")
            f.write(text + "\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"Temperature comparison saved to {save_path}")


def analyze_character_distribution(text, save_path='char_distribution.png'):
    """
    Analyze and plot character distribution in the text.
    
    Args:
        text: Input text string
        save_path: Path to save the plot
    """
    from collections import Counter
    
    # Count characters
    char_counts = Counter(text)
    
    # Get top 30 most common characters
    most_common = char_counts.most_common(30)
    chars, counts = zip(*most_common)
    
    # Create plot
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(chars)), counts, color='steelblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Character', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Top 30 Most Frequent Characters in Dataset', fontsize=14, fontweight='bold')
    plt.xticks(range(len(chars)), [repr(c) for c in chars], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Character distribution plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("This module provides visualization utilities.")
    print("Import and use the functions in your training scripts.")
