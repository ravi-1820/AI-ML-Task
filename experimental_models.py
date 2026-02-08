# LSTM Text Generator - Experimental Variations

This script explores different LSTM architectures and hyperparameters
to analyze their impact on text generation quality.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


class ExperimentalLSTMGenerator:
    """
    Experimental LSTM text generator with configurable architectures.
    """
    
    def __init__(self, vocab_size, sequence_length=100):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.model = None
    
    def build_shallow_model(self):
        """Build a shallow LSTM model (1 layer, fewer units)."""
        model = keras.Sequential([
            layers.Embedding(self.vocab_size, 128, input_length=self.sequence_length),
            layers.LSTM(256, dropout=0.2),
            layers.Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_deep_model(self):
        """Build a deep LSTM model (3 layers, more units)."""
        model = keras.Sequential([
            layers.Embedding(self.vocab_size, 512, input_length=self.sequence_length),
            layers.LSTM(1024, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.LSTM(1024, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.LSTM(512, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_bidirectional_model(self):
        """Build a bidirectional LSTM model."""
        model = keras.Sequential([
            layers.Embedding(self.vocab_size, 256, input_length=self.sequence_length),
            layers.Bidirectional(layers.LSTM(512, return_sequences=True, dropout=0.2)),
            layers.Bidirectional(layers.LSTM(256, dropout=0.2)),
            layers.Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_gru_model(self):
        """Build a GRU-based model (alternative to LSTM)."""
        model = keras.Sequential([
            layers.Embedding(self.vocab_size, 256, input_length=self.sequence_length),
            layers.GRU(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.GRU(512, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        self.model = model
        return model


def compare_architectures(X_train, y_train, X_val, y_val, vocab_size, sequence_length):
    """
    Compare different model architectures.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        vocab_size: Size of character vocabulary
        sequence_length: Length of input sequences
    """
    architectures = {
        'Shallow LSTM': 'build_shallow_model',
        'Deep LSTM': 'build_deep_model',
        'Bidirectional LSTM': 'build_bidirectional_model',
        'GRU': 'build_gru_model'
    }
    
    results = {}
    
    for name, method_name in architectures.items():
        print(f"\n{'='*80}")
        print(f"Training: {name}")
        print(f"{'='*80}")
        
        generator = ExperimentalLSTMGenerator(vocab_size, sequence_length)
        method = getattr(generator, method_name)
        model = method()
        
        print(f"\nModel Summary for {name}:")
        model.summary()
        
        # Train for fewer epochs for comparison
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=128,
            verbose=1
        )
        
        # Save results
        results[name] = {
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1]
        }
        
        # Save model
        model.save(f'{name.lower().replace(" ", "_")}_model.keras')
    
    # Print comparison
    print(f"\n{'='*80}")
    print("ARCHITECTURE COMPARISON RESULTS")
    print(f"{'='*80}")
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Training Loss: {metrics['final_loss']:.4f}")
        print(f"  Validation Loss: {metrics['final_val_loss']:.4f}")
        print(f"  Training Accuracy: {metrics['final_accuracy']:.4f}")
        print(f"  Validation Accuracy: {metrics['final_val_accuracy']:.4f}")
    
    return results


if __name__ == "__main__":
    print("This script is meant to be imported and used with the main generator.")
    print("See the README for usage instructions.")
