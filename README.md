# LSTM Text Generator

A comprehensive implementation of a character-level LSTM text generator trained on Shakespeare's works.

## 📋 Overview

This project implements a Long Short-Term Memory (LSTM) neural network for text generation. The model learns patterns from Shakespeare's complete works and generates new text in a similar style.

## 🎯 Features

- **Character-level text generation**: Learns patterns at the character level for more flexible generation
- **Configurable architecture**: Adjustable LSTM layers, embedding dimensions, and sequence lengths
- **Multiple model variants**: Includes experimental architectures (shallow, deep, bidirectional, GRU)
- **Temperature-based sampling**: Control randomness in text generation
- **Automatic dataset download**: Downloads Shakespeare's works from Project Gutenberg
- **Model checkpointing**: Saves best model during training
- **Early stopping**: Prevents overfitting

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Dependencies:
- TensorFlow >= 2.13.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- Requests >= 2.31.0

## 🚀 Quick Start

### Basic Usage

Run the main script to train and generate text:

```bash
python lstm_text_generator.py
```

This will:
1. Download Shakespeare's complete works
2. Preprocess the text
3. Train the LSTM model
4. Generate sample texts with different seeds and temperatures

### Custom Training

```python
from lstm_text_generator import LSTMTextGenerator

# Initialize generator
generator = LSTMTextGenerator(
    sequence_length=100,    # Length of input sequences
    embedding_dim=256,      # Embedding dimension
    lstm_units=512          # LSTM units
)

# Load and preprocess data
text = generator.load_and_preprocess_text('your_text.txt')
generator.create_char_mappings(text)

# Prepare sequences
X_train, y_train, X_val, y_val = generator.prepare_sequences(text)

# Build and train
generator.build_model(lstm_layers=2, dropout_rate=0.2)
generator.train(X_train, y_train, X_val, y_val, epochs=30)

# Generate text
generated = generator.generate_text(
    seed_text="to be or not to be",
    length=500,
    temperature=1.0
)
```

## 🔬 Experimental Models

Compare different architectures:

```python
from experimental_models import compare_architectures

results = compare_architectures(
    X_train, y_train, X_val, y_val,
    vocab_size=generator.vocab_size,
    sequence_length=100
)
```

Available architectures:
- **Shallow LSTM**: 1 layer, 256 units (faster training, simpler patterns)
- **Deep LSTM**: 3 layers, 1024 units (better quality, slower training)
- **Bidirectional LSTM**: Processes sequences in both directions
- **GRU**: Alternative to LSTM, often faster with similar performance

## 📊 Dataset

**Default Dataset**: Shakespeare's Complete Works from Project Gutenberg
- **Source**: https://www.gutenberg.org/files/100/100-0.txt
- **Size**: ~5.5 MB of text
- **Format**: Plain text (.txt)

### Using Custom Datasets

You can use any text dataset:

```python
generator.load_and_preprocess_text('path/to/your/dataset.txt')
```

**Recommended datasets**:
- [Project Gutenberg](https://www.gutenberg.org/) - Classic literature
- [Kaggle Text Datasets](https://www.kaggle.com/datasets) - Various text corpora
- Any large .txt file with consistent style

## 🎨 Text Generation Parameters

### Temperature

Controls randomness in generation:
- **0.5**: Conservative, more predictable text
- **1.0**: Balanced creativity and coherence
- **1.5**: More creative, less predictable

```python
# Conservative generation
generator.generate_text("to be", length=300, temperature=0.5)

# Creative generation
generator.generate_text("to be", length=300, temperature=1.5)
```

### Seed Text

Starting text for generation:
- Should be at least a few words
- Will be preprocessed (lowercased, special chars removed)
- Longer seeds often produce more coherent results

## 📈 Model Architecture

### Default Configuration

```
Input (sequence_length=100)
    ↓
Embedding Layer (256 dimensions)
    ↓
LSTM Layer 1 (512 units, return_sequences=True)
    ↓
LSTM Layer 2 (512 units)
    ↓
Dense Layer (vocab_size, softmax)
    ↓
Output (next character prediction)
```

### Hyperparameters

- **Sequence Length**: 100 characters
- **Embedding Dimension**: 256
- **LSTM Units**: 512
- **LSTM Layers**: 2
- **Dropout Rate**: 0.2
- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

## 📁 Project Structure

```
lstm_text_generator/
├── lstm_text_generator.py    # Main implementation
├── experimental_models.py     # Alternative architectures
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── shakespeare.txt            # Downloaded dataset (auto-generated)
├── best_model.keras          # Best model checkpoint (auto-generated)
└── lstm_text_generator.keras # Final trained model (auto-generated)
```

## 🎯 Sample Output

### Seed: "to be or not to be"
**Temperature 0.5** (Conservative):
```
to be or not to be the world and the world and the world
and the world and the world and the world...
```

**Temperature 1.0** (Balanced):
```
to be or not to be a man of the world that shall be
the heart of the world, and the world shall see...
```

**Temperature 1.5** (Creative):
```
to be or not to be strange fortunes, my lord, what
dost thou speak of love and death in thy heart...
```

## 🔧 Troubleshooting

### Out of Memory Error
- Reduce `batch_size` (try 64 or 32)
- Reduce `lstm_units` (try 256)
- Reduce `sequence_length` (try 50)

### Poor Text Quality
- Train for more epochs (try 50-100)
- Increase model complexity (more layers/units)
- Use a larger dataset
- Adjust temperature during generation

### Slow Training
- Use GPU acceleration (install tensorflow-gpu)
- Reduce model complexity
- Reduce batch size
- Use GRU instead of LSTM

## 📝 Evaluation Criteria

### Model Performance
- **Coherence**: Generated text should be readable
- **Style matching**: Should resemble training data style
- **Diversity**: Should generate varied outputs with different seeds

### Code Quality
- Well-documented with clear comments
- Modular and reusable design
- Follows Python best practices
- Error handling and validation

### Experimentation
- Multiple architecture comparisons
- Hyperparameter tuning
- Temperature analysis
- Different sequence lengths

## 🎓 Learning Outcomes

This project demonstrates:
1. **Text preprocessing**: Tokenization, sequence creation
2. **LSTM architecture**: Embedding, recurrent layers, output layers
3. **Training techniques**: Checkpointing, early stopping, validation
4. **Text generation**: Sampling strategies, temperature control
5. **Model comparison**: Evaluating different architectures

## 📚 References

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [TensorFlow Text Generation Tutorial](https://www.tensorflow.org/text/tutorials/text_generation)

## 📄 License

This project is for educational purposes. The Shakespeare dataset is in the public domain.

## 👤 Author

AI/ML Project - February 2026

---

**Happy Text Generating! 🎭**
