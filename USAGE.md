# LSTM Text Generator - Usage Guide

## Quick Start Options

### Option 1: Run the Main Script (Full Training)
```bash
python lstm_text_generator.py
```
This will:
- Download Shakespeare's works
- Train the model for 30 epochs
- Generate sample texts with multiple seeds and temperatures

### Option 2: Quick Start (Simplified)
```bash
python quick_start.py
```
This will:
- Train for 20 epochs (faster)
- Generate visualizations
- Create sample outputs

### Option 3: Step-by-Step Tutorial
```bash
python tutorial.py
```
This provides a detailed walkthrough of each step.

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify installation:**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"
```

## Training Your Own Model

### Basic Training
```python
from lstm_text_generator import LSTMTextGenerator

# Initialize
generator = LSTMTextGenerator(
    sequence_length=100,
    embedding_dim=256,
    lstm_units=512
)

# Load data
text = generator.load_and_preprocess_text('your_text.txt')
generator.create_char_mappings(text)

# Prepare sequences
X_train, y_train, X_val, y_val = generator.prepare_sequences(text)

# Build and train
generator.build_model(lstm_layers=2, dropout_rate=0.2)
history = generator.train(X_train, y_train, X_val, y_val, epochs=30)

# Save
generator.save_model('my_model.keras')
```

### Generate Text
```python
# Generate with different temperatures
generator.generate_text("to be or not to be", length=500, temperature=0.5)  # Conservative
generator.generate_text("to be or not to be", length=500, temperature=1.0)  # Balanced
generator.generate_text("to be or not to be", length=500, temperature=1.5)  # Creative
```

## Experimenting with Architectures

```python
from experimental_models import compare_architectures

# Compare different architectures
results = compare_architectures(
    X_train, y_train, X_val, y_val,
    vocab_size=generator.vocab_size,
    sequence_length=100
)
```

This will train and compare:
- Shallow LSTM (1 layer, 256 units)
- Deep LSTM (3 layers, 1024 units)
- Bidirectional LSTM
- GRU-based model

## Visualization

```python
from visualizations import plot_training_history, analyze_character_distribution

# Plot training metrics
plot_training_history(history, 'training_plot.png')

# Analyze dataset
analyze_character_distribution(text, 'char_dist.png')
```

## Tips for Better Results

### 1. Increase Training Time
```python
generator.train(X_train, y_train, X_val, y_val, epochs=50)  # More epochs
```

### 2. Adjust Model Complexity
```python
# Larger model
generator = LSTMTextGenerator(
    sequence_length=150,      # Longer sequences
    embedding_dim=512,        # Larger embeddings
    lstm_units=1024          # More LSTM units
)
```

### 3. Use Different Datasets
- Poetry collections
- Code repositories
- News articles
- Any large text corpus

### 4. Fine-tune Temperature
- **0.2-0.5**: Very conservative, repetitive
- **0.5-0.8**: Coherent but predictable
- **0.8-1.2**: Good balance
- **1.2-1.5**: Creative but may lose coherence
- **1.5+**: Very random

## Troubleshooting

### Memory Issues
```python
# Reduce batch size
generator.train(X_train, y_train, X_val, y_val, batch_size=64)

# Or reduce model size
generator = LSTMTextGenerator(lstm_units=256)
```

### Slow Training
```python
# Use GPU if available
print(tf.config.list_physical_devices('GPU'))

# Or reduce complexity
generator.build_model(lstm_layers=1)
```

### Poor Quality Output
- Train longer (50+ epochs)
- Use larger dataset
- Increase model complexity
- Adjust temperature during generation

## Expected Training Time

On CPU:
- Small model (256 units, 1 layer): ~2-3 hours for 30 epochs
- Medium model (512 units, 2 layers): ~4-6 hours for 30 epochs
- Large model (1024 units, 3 layers): ~8-12 hours for 30 epochs

On GPU:
- 5-10x faster than CPU

## Project Files

After running, you'll have:
```
lstm_text_generator/
├── lstm_text_generator.py       # Main implementation
├── experimental_models.py        # Alternative architectures
├── visualizations.py             # Plotting utilities
├── quick_start.py               # Simplified interface
├── tutorial.py                  # Step-by-step guide
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
├── USAGE.md                     # This file
├── shakespeare.txt              # Downloaded dataset
├── best_model.keras            # Best checkpoint
├── lstm_text_generator.keras   # Final model
├── training_history.png        # Training plots
└── char_distribution.png       # Character analysis
```

## Next Steps

1. **Run the quick start** to see it in action
2. **Experiment with temperatures** to find the sweet spot
3. **Try different architectures** to compare performance
4. **Use your own dataset** for custom text generation
5. **Tune hyperparameters** for better results

Happy text generating! 🎭
