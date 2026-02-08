"""
Quick start script for LSTM text generator.
This script provides a simplified interface for training and generating text.
"""

from lstm_text_generator import LSTMTextGenerator
from visualizations import plot_training_history, analyze_character_distribution
import os


def quick_train(dataset_path='shakespeare.txt', epochs=20, sequence_length=100):
    """
    Quick training with default parameters.
    
    Args:
        dataset_path: Path to text dataset
        epochs: Number of training epochs
        sequence_length: Length of input sequences
    """
    print("=" * 80)
    print("QUICK START - LSTM TEXT GENERATOR")
    print("=" * 80)
    
    # Initialize generator
    generator = LSTMTextGenerator(
        sequence_length=sequence_length,
        embedding_dim=256,
        lstm_units=512
    )
    
    # Download dataset if needed
    if not os.path.exists(dataset_path):
        print("\nDataset not found. Downloading Shakespeare's works...")
        generator.download_dataset(save_path=dataset_path)
    
    # Load and preprocess
    print("\nLoading and preprocessing text...")
    text = generator.load_and_preprocess_text(dataset_path)
    
    # Analyze character distribution
    print("\nAnalyzing character distribution...")
    analyze_character_distribution(text, 'char_distribution.png')
    
    # Create mappings
    generator.create_char_mappings(text)
    
    # Prepare sequences
    print("\nPreparing sequences...")
    X_train, y_train, X_val, y_val = generator.prepare_sequences(text, validation_split=0.1)
    
    # Build model
    print("\nBuilding model...")
    generator.build_model(lstm_layers=2, dropout_rate=0.2)
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    history = generator.train(
        X_train, y_train, X_val, y_val,
        epochs=epochs,
        batch_size=128
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history, 'training_history.png')
    
    # Save model
    generator.save_model('lstm_text_generator.keras')
    
    return generator


def quick_generate(generator, seed_texts=None, temperatures=None):
    """
    Quick text generation with default seeds.
    
    Args:
        generator: Trained LSTMTextGenerator instance
        seed_texts: List of seed texts (optional)
        temperatures: List of temperatures (optional)
    """
    if seed_texts is None:
        seed_texts = [
            "to be or not to be",
            "all the world's a stage",
            "shall i compare thee"
        ]
    
    if temperatures is None:
        temperatures = [0.5, 1.0, 1.5]
    
    print("\n" + "=" * 80)
    print("GENERATING TEXT SAMPLES")
    print("=" * 80)
    
    for seed in seed_texts:
        print(f"\n\nSeed: '{seed}'")
        print("-" * 80)
        
        for temp in temperatures:
            print(f"\nTemperature: {temp}")
            generator.generate_text(seed, length=200, temperature=temp)


def main():
    """Main quick start function."""
    # Train the model
    generator = quick_train(epochs=20)
    
    # Generate sample texts
    quick_generate(generator)
    
    print("\n" + "=" * 80)
    print("QUICK START COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - lstm_text_generator.keras (trained model)")
    print("  - best_model.keras (best checkpoint)")
    print("  - training_history.png (training plots)")
    print("  - char_distribution.png (character analysis)")
    print("\nTo generate more text, use:")
    print("  generator.generate_text('your seed text', length=500, temperature=1.0)")


if __name__ == "__main__":
    main()
