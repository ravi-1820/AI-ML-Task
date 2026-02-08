"""
Quick start script for LSTM text generator - LITE VERSION
This script is optimized for systems with limited RAM.
"""

from lstm_text_generator import LSTMTextGenerator
from visualizations import plot_training_history, analyze_character_distribution
import os


def quick_train_lite(dataset_path='shakespeare.txt', epochs=15, sequence_length=50):
    """
    Memory-efficient training with reduced parameters.
    
    Args:
        dataset_path: Path to text dataset
        epochs: Number of training epochs
        sequence_length: Length of input sequences (reduced for memory)
    """
    print("=" * 80)
    print("QUICK START LITE - LSTM TEXT GENERATOR (Memory Optimized)")
    print("=" * 80)
    
    # Initialize generator with SMALLER parameters
    generator = LSTMTextGenerator(
        sequence_length=sequence_length,  # 50 instead of 100
        embedding_dim=128,                # 128 instead of 256
        lstm_units=256                    # 256 instead of 512
    )
    
    # Download dataset if needed
    if not os.path.exists(dataset_path):
        print("\nDataset not found. Downloading Shakespeare's works...")
        generator.download_dataset(save_path=dataset_path)
    
    # Load and preprocess
    print("\nLoading and preprocessing text...")
    text = generator.load_and_preprocess_text(dataset_path)
    
    # Use only FIRST 500,000 characters to reduce memory usage
    if len(text) > 500000:
        print(f"\nUsing first 500,000 characters (out of {len(text)}) to save memory...")
        text = text[:500000]
    
    # Analyze character distribution
    print("\nAnalyzing character distribution...")
    try:
        analyze_character_distribution(text, 'char_distribution.png')
    except:
        print("Skipping visualization to save memory")
    
    # Create mappings
    generator.create_char_mappings(text)
    
    # Prepare sequences
    print("\nPreparing sequences...")
    X_train, y_train, X_val, y_val = generator.prepare_sequences(text, validation_split=0.1)
    
    # Build SMALLER model (only 1 LSTM layer)
    print("\nBuilding memory-efficient model...")
    generator.build_model(lstm_layers=1, dropout_rate=0.2)  # Only 1 layer!
    
    # Train with SMALLER batch size
    print(f"\nTraining for {epochs} epochs with reduced batch size...")
    history = generator.train(
        X_train, y_train, X_val, y_val,
        epochs=epochs,
        batch_size=32  # 32 instead of 128 - MUCH less memory!
    )
    
    # Plot training history
    print("\nPlotting training history...")
    try:
        plot_training_history(history, 'training_history.png')
    except:
        print("Skipping plot to save memory")
    
    # Save model
    generator.save_model('lstm_text_generator_lite.keras')
    
    return generator


def quick_generate(generator, seed_texts=None, temperatures=None):
    """
    Quick text generation with default seeds.
    """
    if seed_texts is None:
        seed_texts = [
            "to be or not to be",
            "all the world's a stage"
        ]
    
    if temperatures is None:
        temperatures = [0.8, 1.0, 1.2]
    
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
    """Main quick start function - LITE VERSION."""
    # Train the model with reduced parameters
    generator = quick_train_lite(epochs=15)
    
    # Generate sample texts
    quick_generate(generator)
    
    print("\n" + "=" * 80)
    print("QUICK START LITE COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - lstm_text_generator_lite.keras (trained model)")
    print("  - best_model.keras (best checkpoint)")
    print("\nModel specs (memory-optimized):")
    print("  - Sequence length: 50 (instead of 100)")
    print("  - Embedding dim: 128 (instead of 256)")
    print("  - LSTM units: 256 (instead of 512)")
    print("  - LSTM layers: 1 (instead of 2)")
    print("  - Batch size: 32 (instead of 128)")
    print("\nTo generate more text, use:")
    print("  generator.generate_text('your seed text', length=500, temperature=1.0)")


if __name__ == "__main__":
    main()
