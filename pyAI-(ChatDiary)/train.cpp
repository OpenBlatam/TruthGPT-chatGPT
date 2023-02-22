void AutoregressiveTransformer::train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y, int epochs) {
    int num_batches = (int)X.size() / batch_size;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double loss = 0.0;

        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            int start_idx = batch_idx * batch_size;
            int end_idx = start_idx + batch_size;

            std::vector<std::vector<double>> batch_X(X.begin() + start_idx, X.begin() + end_idx);
            std::vector<std::vector<double>> batch_Y(Y.begin() + start_idx, Y.begin() + end_idx);

            // Compute the gradients for the current batch
            for (int i = 0; i < batch_size; i++) {
                std::vector<double> x = batch_X[i];
                std::vector<double> y = batch_Y[i];

                // Forward pass
                std::vector<double> y_pred = forward(x);

                // Compute the loss and accumulate it
                double batch_loss = cross_entropy_loss(y_pred, y);
                loss += batch_loss;

                // Backward pass
                std::vector<double> grad = softmax_cross_entropy_loss_gradient(y_pred, y);
                backward(x, grad);
            }

            // Update the model parameters after processing the batch
            update_parameters();
        }

        // Compute the average loss for the current epoch
        loss /= X.size();

        std::cout << "Epoch " << epoch << " loss: " << loss << std::endl;
    }
}
