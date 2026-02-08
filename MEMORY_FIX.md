# Memory Error Fix Guide (Hinglish)

## 🚨 Problem: OOM (Out of Memory) Error

Agar aapko **"OOM when allocating tensor"** error aa raha hai, iska matlab hai ki aapke system mein RAM kam hai training ke liye.

## ✅ Solution: Lite Version Use Karo

### **Quick Fix - Abhi Run Karo:**

```bash
python quick_start_lite.py
```

Ye **memory-optimized version** hai jo kam RAM use karega!

---

## 🔧 Kya Changes Kiye Gaye

| Parameter | Original | Lite Version | Memory Saved |
|-----------|----------|--------------|--------------|
| Sequence Length | 100 | 50 | 50% less |
| Embedding Dim | 256 | 128 | 50% less |
| LSTM Units | 512 | 256 | 50% less |
| LSTM Layers | 2 | 1 | 50% less |
| Batch Size | 128 | **32** | **75% less** |
| Dataset Size | Full (~5.5M) | 500K chars | 90% less |

**Result:** Lagbhag **80-90% kam memory** use hogi! 🎉

---

## 📊 Performance Comparison

### Original Version:
- **RAM Required:** ~8-12 GB
- **Training Time:** 4-6 hours
- **Quality:** Excellent

### Lite Version:
- **RAM Required:** ~2-4 GB ✅
- **Training Time:** 1-2 hours ✅
- **Quality:** Good (thoda kam but still usable)

---

## 🎯 Step-by-Step Instructions

### **Option 1: Lite Version (Recommended)**

```bash
cd c:\Users\rajja\OneDrive\Desktop\AI_ML\lstm_text_generator
python quick_start_lite.py
```

**Ye karega:**
- Chhota model banayega
- Kam data use karega
- Kam memory use karega
- 15 epochs train karega
- 1-2 ghante mein complete hoga

---

### **Option 2: Manual Fix (Agar Lite Version Bhi Fail Ho)**

Agar lite version bhi memory error de, toh aur bhi chhota karo:

```python
from lstm_text_generator import LSTMTextGenerator

# BAHUT CHHOTA MODEL
generator = LSTMTextGenerator(
    sequence_length=30,      # Aur chhota
    embedding_dim=64,        # Aur chhota
    lstm_units=128           # Aur chhota
)

# Load text
text = generator.load_and_preprocess_text('shakespeare.txt')
text = text[:200000]  # Sirf 200K characters

# Create mappings
generator.create_char_mappings(text)

# Prepare sequences
X_train, y_train, X_val, y_val = generator.prepare_sequences(text)

# Build model with 1 layer only
generator.build_model(lstm_layers=1, dropout_rate=0.1)

# Train with VERY small batch
generator.train(
    X_train, y_train, X_val, y_val,
    epochs=10,
    batch_size=16  # Bahut chhoti batch!
)

# Generate
generator.generate_text("to be", length=200, temperature=1.0)
```

---

## 💡 Extra Tips

### **Agar Phir Bhi Error Aaye:**

1. **Unnecessary programs band karo:**
   - Chrome/Browser tabs
   - Other heavy applications
   - Background processes

2. **System restart karo:**
   - Fresh RAM milega

3. **Batch size aur chhoti karo:**
   ```python
   batch_size=8  # Ya 16
   ```

4. **Dataset aur chhota karo:**
   ```python
   text = text[:100000]  # Sirf 100K characters
   ```

---

## 🎭 Quality vs Memory Trade-off

| Setting | Memory | Quality | Speed |
|---------|--------|---------|-------|
| Full Version | High | Best | Slow |
| Lite Version | Medium | Good | Medium |
| Ultra-Lite | Low | OK | Fast |

**Recommendation:** Lite version se shuru karo. Agar chal gaya, perfect! Agar nahi, toh ultra-lite try karo.

---

## ✅ Success Indicators

Agar ye dikhe toh sab theek hai:
```
Epoch 1/15
1234/1234 [==============================] - 45s 36ms/step - loss: 2.1234 - accuracy: 0.3456
```

Agar phir se OOM error aaye:
```
OOM when allocating tensor...
```
Toh batch_size aur chhoti karo (16 ya 8).

---

## 🚀 Final Command

**Sabse safe option:**
```bash
python quick_start_lite.py
```

**Agar ye bhi fail ho:**
Manual fix use karo with batch_size=16

Good luck! 💪
