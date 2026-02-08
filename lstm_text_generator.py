"""
LSTM Text Generator
===================
This script implements a character-level LSTM text generator trained on Shakespeare's works.

Author: AI/ML Project
Date: February 2026
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import requests
import re


class LSTMTextGenerator:
    """
    A character-level LSTM text generator that learns to generate text
    in the style of the training corpus.
    """
    
    def __init__(self, sequence_length=100, embedding_dim=256, lstm_units=512):
        """
        Initialize the text generator.
        
        Args:
            sequence_length (int): Length of input sequences for training
            embedding_dim (int): Dimension of character embeddings
            lstm_units (int): Number of units in LSTM layers
        """
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
    def download_dataset(self, url=None, save_path='shakespeare.txt'):
        """
        Download the Shakespeare dataset from Project Gutenberg.
        
        Args:
            url (str): URL to download from (default: Shakespeare's complete works)
            save_path (str): Path to save the downloaded text
            
        Returns:
            str: Path to the downloaded file
        """
        if url is None:
            # Shakespeare's complete works from Project Gutenberg
            url = 'https://www.gutenberg.org/files/100/100-0.txt'
        
        print(f"Downloading dataset from {url}...")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"Dataset downloaded successfully to {save_path}")
            return save_path
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise
    
    def load_and_preprocess_text(self, file_path):
        """
        Load and preprocess the text data.
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            str: Preprocessed text
        """
        print(f"Loading text from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        # Keep letters, numbers, spaces, and common punctuation
        text = re.sub(r'[^a-z0-9\s.,!?;:\'\-\n]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        print(f"Text loaded. Length: {len(text)} characters")
        
        return text
    
    def create_char_mappings(self, text):
        """
        Create character to index and index to character mappings.
        
        Args:
            text (str): Input text
        """
        # Get unique characters
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(chars)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Unique characters: {''.join(chars[:50])}...")
    
    def prepare_sequences(self, text, validation_split=0.1):
        """
        Prepare input-output sequences for training.
        
        Args:
            text (str): Input text
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            tuple: (X_train, y_train, X_val, y_val)
        """
        print("Preparing sequences...")
        
        # Convert text to indices
        text_indices = [self.char_to_idx[char] for char in text]
        
        # Create sequences
        sequences = []
        next_chars = []
        
        for i in range(len(text_indices) - self.sequence_length):
            sequences.append(text_indices[i:i + self.sequence_length])
            next_chars.append(text_indices[i + self.sequence_length])
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y = np.array(next_chars)
        
        # Convert y to categorical (one-hot encoding)
        y = keras.utils.to_categorical(y, num_classes=self.vocab_size)
        
        # Split into training and validation sets
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Training sequences: {len(X_train)}")
        print(f"Validation sequences: {len(X_val)}")
        print(f"Input shape: {X_train.shape}")
        print(f"Output shape: {y_train.shape}")
        
        return X_train, y_train, X_val, y_val
    
    def build_model(self, lstm_layers=2, dropout_rate=0.2):
        """
        Build the LSTM model architecture.
        
        Args:
            lstm_layers (int): Number of LSTM layers
            dropout_rate (float): Dropout rate for regularization
        """
        print("Building model...")
        
        model = keras.Sequential()
        
        # Embedding layer
        model.add(layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.sequence_length
        ))
        
        # LSTM layers
        for i in range(lstm_layers):
            return_sequences = (i < lstm_layers - 1)  # Return sequences for all but last layer
            model.add(layers.LSTM(
                self.lstm_units,
                return_sequences=return_sequences,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate
            ))
        
        # Dense output layer with softmax activation
        model.add(layers.Dense(self.vocab_size, activation='softmax'))
        
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        self.model = model
        
        print("\nModel Summary:")
        model.summary()
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training input sequences
            y_train: Training output sequences
            X_val: Validation input sequences
            y_val: Validation output sequences
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            History: Training history object
        """
        print("\nStarting training...")
        
        # Create callbacks
        checkpoint = ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping],
            verbose=1
        )
        
        print("\nTraining completed!")
        
        return history
    
    def generate_text(self, seed_text, length=500, temperature=1.0):
        """
        Generate text using the trained model.
        
        Args:
            seed_text (str): Initial text to start generation
            length (int): Number of characters to generate
            temperature (float): Sampling temperature (higher = more random)
            
        Returns:
            str: Generated text
        """
        # Preprocess seed text
        seed_text = seed_text.lower()
        seed_text = re.sub(r'[^a-z0-9\s.,!?;:\'\-\n]', '', seed_text)
        
        # Ensure seed text is long enough
        if len(seed_text) < self.sequence_length:
            seed_text = seed_text + ' ' * (self.sequence_length - len(seed_text))
        
        # Take only the last sequence_length characters
        seed_text = seed_text[-self.sequence_length:]
        
        generated_text = seed_text
        
        print(f"\nGenerating text with seed: '{seed_text[:50]}...'")
        print(f"Temperature: {temperature}\n")
        print("-" * 80)
        
        for _ in range(length):
            # Prepare input sequence
            sequence = [self.char_to_idx.get(char, 0) for char in generated_text[-self.sequence_length:]]
            sequence = np.array([sequence])
            
            # Predict next character
            predictions = self.model.predict(sequence, verbose=0)[0]
            
            # Apply temperature
            predictions = np.log(predictions + 1e-10) / temperature
            predictions = np.exp(predictions)
            predictions = predictions / np.sum(predictions)
            
            # Sample next character
            next_idx = np.random.choice(len(predictions), p=predictions)
            next_char = self.idx_to_char[next_idx]
            
            generated_text += next_char
        
        print(generated_text)
        print("-" * 80)
        
        return generated_text
    
    def save_model(self, path='lstm_text_generator.keras'):
        """Save the trained model."""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='lstm_text_generator.keras'):
        """Load a trained model."""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")


def main():
    """
    Main function to run the complete text generation pipeline.
    """
    print("=" * 80)
    print("LSTM Text Generator - Shakespeare Edition")
    print("=" * 80)
    
    # Initialize the generator
    generator = LSTMTextGenerator(
        sequence_length=100,
        embedding_dim=256,
        lstm_units=512
    )
    
    # Download dataset
    dataset_path = 'shakespeare.txt'
    if not os.path.exists(dataset_path):
        generator.download_dataset(save_path=dataset_path)
    else:
        print(f"Using existing dataset: {dataset_path}")
    
    # Load and preprocess text
    text = generator.load_and_preprocess_text(dataset_path)
    
    # Create character mappings
    generator.create_char_mappings(text)
    
    # Prepare sequences
    X_train, y_train, X_val, y_val = generator.prepare_sequences(text, validation_split=0.1)
    
    # Build model
    generator.build_model(lstm_layers=2, dropout_rate=0.2)
    
    # Train model
    history = generator.train(
        X_train, y_train, X_val, y_val,
        epochs=30,
        batch_size=128
    )
    
    # Save the model
    generator.save_model('lstm_text_generator.keras')
    
    # Generate sample texts with different seeds and temperatures
    print("\n" + "=" * 80)
    print("GENERATING SAMPLE TEXTS")
    print("=" * 80)
    
    seeds = [
        "to be or not to be",
        "all the world's a stage",
        "shall i compare thee"
    ]
    
    temperatures = [0.5, 1.0, 1.5]
    
    for seed in seeds:
        for temp in temperatures:
            print(f"\n\nSeed: '{seed}' | Temperature: {temp}")
            generator.generate_text(seed, length=300, temperature=temp)
    
    print("\n" + "=" * 80)
    print("Text generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
