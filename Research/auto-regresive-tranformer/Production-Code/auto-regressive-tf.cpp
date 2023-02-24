#include <torch/torch.h>
#include <torch/cuda.h>
#include <vector>
#include <iostream>
#include "autoreg_transformer.h"

int main() {
    // Define hyperparameters
    int num_layers = 6;
    int input_size = 100;
    int hidden_size = 512;
    int output_size = 100;
    float learning_rate = 0.0005;
    int batch_size = 64;
    int num_epochs = 10;

    // Load training data
    std::vector<std::vector<int>> input_seqs, output_seqs;
    // Load training data...

    // Initialize autoregressive transformer model
    AutoRegTransformer model(num_layers, input_size, hidden_size, output_size);

    // Initialize optimizer
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(learning_rate));

    // Enable mixed precision training
    model.to(torch::kHalf);
    optimizer = torch::optim::Adam(model.parameters(), torch::optim::AdamOptions(learning_rate).set_dtype(torch::kHalf));

    // Train model for multiple epochs
    for (int epoch = 1; epoch <= num_epochs; epoch++) {
        // Iterate over training data
        float epoch_loss = train(model, input_seqs, output_seqs, optimizer, batch_size);

        std::cout << "Epoch " << epoch << " loss: " << epoch_loss << std::endl;
    }

    return 0;
}
