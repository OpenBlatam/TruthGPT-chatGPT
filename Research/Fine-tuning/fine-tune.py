def fine_tuned_model(base_model_output):
    x = Dense(128, activation='relu')(base_model_output)  # Add a dense layer with 128 units and ReLU activation
    x = BatchNormalization()(x)  # Apply batch normalization
    output = Dense(10, activation='softmax')(x)  # Output layer with 10 units for classification

    return output
