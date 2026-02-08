# LSTM Text Generator - Getting Started

## 🚀 Installation & Setup

### Step 1: Install Dependencies
```bash
cd c:\Users\rajja\OneDrive\Desktop\AI_ML\lstm_text_generator
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed successfully')"
```

## 🎯 Three Ways to Run

### Option 1: Quick Start (Recommended for First Time)
```bash
python quick_start.py
```
**What it does:**
- Trains for 20 epochs (~2-4 hours on CPU)
- Creates visualizations automatically
- Generates sample texts
- Perfect for testing the system

### Option 2: Full Training
```bash
python lstm_text_generator.py
```
**What it does:**
- Trains for 30 epochs (~4-6 hours on CPU)
- Generates extensive samples
- Tests multiple temperatures
- Production-quality results

### Option 3: Tutorial Mode
```bash
python tutorial.py
```
**What it does:**
- Step-by-step walkthrough
- Educational explanations
- Shows intermediate results
- Great for learning

## 📝 What You'll Get

After running, you'll have:
- ✅ `shakespeare.txt` - Downloaded dataset
- ✅ `best_model.keras` - Best model checkpoint
- ✅ `lstm_text_generator.keras` - Final trained model
- ✅ `training_history.png` - Training plots (quick_start only)
- ✅ `char_distribution.png` - Character analysis (quick_start only)

## 🎨 Generate Your Own Text

After training, use the model:

```python
from lstm_text_generator import LSTMTextGenerator

# Load trained model
generator = LSTMTextGenerator()
generator.load_model('lstm_text_generator.keras')

# Generate text
generator.generate_text(
    seed_text="your seed text here",
    length=500,
    temperature=1.0
)
```

## 🔧 Customize Your Model

Edit these parameters in the script:

```python
generator = LSTMTextGenerator(
    sequence_length=100,    # Try 50, 100, or 150
    embedding_dim=256,      # Try 128, 256, or 512
    lstm_units=512          # Try 256, 512, or 1024
)

generator.build_model(
    lstm_layers=2,          # Try 1, 2, or 3
    dropout_rate=0.2        # Try 0.1, 0.2, or 0.3
)

generator.train(
    epochs=30,              # Try 20, 30, or 50
    batch_size=128          # Try 64, 128, or 256
)
```

## ⚡ Performance Tips

### If Training is Too Slow:
- Reduce `lstm_units` to 256
- Reduce `batch_size` to 64
- Use 1 LSTM layer instead of 2
- Train for fewer epochs (20 instead of 30)

### If Running Out of Memory:
- Reduce `batch_size` to 32
- Reduce `lstm_units` to 256
- Reduce `sequence_length` to 50

### For Better Quality:
- Train for more epochs (50+)
- Increase `lstm_units` to 1024
- Use 3 LSTM layers
- Increase `sequence_length` to 150

## 🎭 Temperature Guide

| Temperature | Result | When to Use |
|------------|--------|-------------|
| 0.2-0.5 | Very predictable, repetitive | Testing, consistency |
| 0.5-0.8 | Coherent, safe | General use |
| 0.8-1.2 | **Balanced (recommended)** | Best quality |
| 1.2-1.5 | Creative, varied | Experimentation |
| 1.5+ | Random, chaotic | Fun experiments |

## 📚 Next Steps

1. **Start with quick_start.py** to see it work
2. **Experiment with temperatures** to find your preference
3. **Try your own dataset** for custom text generation
4. **Compare architectures** using experimental_models.py
5. **Tune hyperparameters** for better results

## ❓ Need Help?

- Check [README.md](README.md) for detailed documentation
- Check [USAGE.md](USAGE.md) for practical examples
- Check [walkthrough.md](file:///C:/Users/rajja/.gemini/antigravity/brain/66c097c3-f3d3-4dcd-8491-881f1857fc4f/walkthrough.md) for project details

---

**Ready to start? Run:** `python quick_start.py`
