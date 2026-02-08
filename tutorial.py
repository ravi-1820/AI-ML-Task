"""
Step-by-step tutorial notebook for LSTM Text Generator.
This file can be run as a Python script or converted to Jupyter notebook.
"""

# %% [markdown]
# # LSTM Text Generator - Step by Step Tutorial
# 
# This tutorial walks through building a text generator using LSTM networks.
# We'll train on Shakespeare's works and generate new text in similar style.

# %% [markdown]
# ## Step 1: Import Libraries

# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import requests
import re
import os

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# %% [markdown]
# ## Step 2: Download and Load Dataset

# %%
def download_shakespeare():
    """Download Shakespeare's complete works."""
    url = 'https://www.gutenberg.org/files/100/100-0.txt'
    
    if not os.path.exists('shakespeare.txt'):
        print("Downloading Shakespeare's works...")
        response = requests.get(url)
        with open('shakespeare.txt', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Download complete!")
    else:
        print("Dataset already exists.")

download_shakespeare()

# Load the text
with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Total characters: {len(text)}")
print(f"First 500 characters:\n{text[:500]}")

# %% [markdown]
# ## Step 3: Preprocess the Text

# %%
# Convert to lowercase
text = text.lower()

# Remove special characters (keep basic punctuation)
text = re.sub(r'[^a-z0-9\s.,!?;:\'\-\n]', '', text)

# Remove extra whitespace
text = re.sub(r'\s+', ' ', text)

print(f"Preprocessed length: {len(text)}")
print(f"Sample: {text[10000:10500]}")

# %% [markdown]
# ## Step 4: Create Character Mappings

# %%
# Get unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {''.join(chars)}")

# Create mappings
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

# Convert text to indices
text_indices = [char_to_idx[char] for char in text]

print(f"Text as indices (first 50): {text_indices[:50]}")

# %% [markdown]
# ## Step 5: Prepare Training Sequences

# %%
sequence_length = 100

# Create input-output pairs
sequences = []
next_chars = []

for i in range(len(text_indices) - sequence_length):
    sequences.append(text_indices[i:i + sequence_length])
    next_chars.append(text_indices[i + sequence_length])

print(f"Total sequences: {len(sequences)}")

# Convert to numpy arrays
X = np.array(sequences)
y = np.array(next_chars)

# One-hot encode the output
y = keras.utils.to_categorical(y, num_classes=vocab_size)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split into train and validation
split_idx = int(len(X) * 0.9)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# %% [markdown]
# ## Step 6: Build the LSTM Model

# %%
# Model parameters
embedding_dim = 256
lstm_units = 512

# Build model
model = keras.Sequential([
    # Embedding layer
    layers.Embedding(vocab_size, embedding_dim, input_length=sequence_length),
    
    # LSTM layers
    layers.LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    layers.LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
    
    # Output layer
    layers.Dense(vocab_size, activation='softmax')
])

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()

# %% [markdown]
# ## Step 7: Train the Model

# %%
# Callbacks
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=128,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

# %% [markdown]
# ## Step 8: Visualize Training Results

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot loss
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# Plot accuracy
axes[1].plot(history.history['accuracy'], label='Training Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_title('Model Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_results.png', dpi=300)
plt.show()

# %% [markdown]
# ## Step 9: Generate Text

# %%
def generate_text(model, seed_text, length=500, temperature=1.0):
    """Generate text using the trained model."""
    
    # Preprocess seed
    seed_text = seed_text.lower()
    seed_text = re.sub(r'[^a-z0-9\s.,!?;:\'\-\n]', '', seed_text)
    
    # Ensure seed is long enough
    if len(seed_text) < sequence_length:
        seed_text = seed_text + ' ' * (sequence_length - len(seed_text))
    
    seed_text = seed_text[-sequence_length:]
    generated = seed_text
    
    print(f"Seed: '{seed_text}'")
    print(f"Temperature: {temperature}\n")
    print("-" * 80)
    
    for _ in range(length):
        # Prepare input
        sequence = [char_to_idx.get(char, 0) for char in generated[-sequence_length:]]
        sequence = np.array([sequence])
        
        # Predict
        predictions = model.predict(sequence, verbose=0)[0]
        
        # Apply temperature
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions)
        predictions = predictions / np.sum(predictions)
        
        # Sample next character
        next_idx = np.random.choice(len(predictions), p=predictions)
        next_char = idx_to_char[next_idx]
        
        generated += next_char
    
    print(generated)
    print("-" * 80)
    
    return generated

# %% [markdown]
# ## Step 10: Generate Sample Texts

# %%
# Test different seeds and temperatures
seeds = [
    "to be or not to be",
    "all the world's a stage",
    "shall i compare thee"
]

temperatures = [0.5, 1.0, 1.5]

for seed in seeds:
    print(f"\n{'='*80}")
    print(f"SEED: '{seed}'")
    print(f"{'='*80}\n")
    
    for temp in temperatures:
        generate_text(model, seed, length=300, temperature=temp)
        print()

# %% [markdown]
# ## Step 11: Save the Model

# %%
model.save('lstm_text_generator_final.keras')
print("Model saved successfully!")

# %% [markdown]
# ## Conclusion
# 
# You've successfully built an LSTM text generator! Key takeaways:
# 
# 1. **Data Preprocessing**: Converting text to sequences is crucial
# 2. **LSTM Architecture**: Embedding + LSTM layers + Dense output
# 3. **Training**: Use callbacks for best results
# 4. **Generation**: Temperature controls creativity vs coherence
# 
# ### Next Steps:
# - Experiment with different architectures (more layers, different units)
# - Try different datasets (poetry, code, etc.)
# - Implement word-level instead of character-level generation
# - Add attention mechanisms for better context

# %%
