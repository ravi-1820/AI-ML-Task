"""
Quick Demo - LSTM Text Generator
This script uses a smaller dataset for faster training (demo purposes).
"""

from lstm_text_generator import LSTMTextGenerator
import os

def main():
    print("=" * 80)
    print("QUICK DEMO - LSTM TEXT GENERATOR")
    print("Using smaller dataset for faster training")
    print("=" * 80)
    
    # Initialize generator with smaller parameters
    generator = LSTMTextGenerator(
        sequence_length=40,  # Shorter sequences
        embedding_dim=128,   # Smaller embedding
        lstm_units=256       # Fewer LSTM units
    )
    
    # Load dataset
    dataset_path = 'shakespeare.txt'
    if not os.path.exists(dataset_path):
        print("\nDataset not found. Downloading...")
        generator.download_dataset(save_path=dataset_path)
    else:
        print(f"\nUsing existing dataset: {dataset_path}")
    
    # Load and preprocess text
    print("\nLoading text...")
    text = generator.load_and_preprocess_text(dataset_path)
    
    # Use only first 100,000 characters for quick demo
    text = text[:100000]
    print(f"Using {len(text)} characters for quick demo")
    
    # Create character mappings
    generator.create_char_mappings(text)
    
    # Prepare sequences
    print("\nPreparing sequences...")
    X_train, y_train, X_val, y_val = generator.prepare_sequences(text, validation_split=0.1)
    
    # Build model with single LSTM layer
    print("\nBuilding model...")
    generator.build_model(lstm_layers=1, dropout_rate=0.2)
    
    # Train for just 5 epochs (quick demo)
    print("\nTraining for 5 epochs (quick demo)...")
    history = generator.train(
        X_train, y_train, X_val, y_val,
        epochs=5,
        batch_size=64
    )
    
    # Save model
    generator.save_model('demo_model.keras')
    
    # Generate sample texts
    print("\n" + "=" * 80)
    print("GENERATING SAMPLE TEXTS")
    print("=" * 80)
    
    seeds = [
        "to be or not to be",
        "all the world",
        "love is"
    ]
    
    for seed in seeds:
        print(f"\n\nSeed: '{seed}'")
        print("-" * 80)
        generator.generate_text(seed, length=200, temperature=1.0)
    
    print("\n" + "=" * 80)
    print("QUICK DEMO COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - demo_model.keras (trained model)")
    print("  - best_model.keras (best checkpoint)")
    print("\nNote: This is a quick demo with limited training.")
    print("For better results, use the full training script.")

if __name__ == "__main__":
    main()
