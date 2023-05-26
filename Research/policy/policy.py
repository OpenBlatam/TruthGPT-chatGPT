import numpy as np

class Policy:
    # ... (rest of the class implementation)

    def optimize_temperature(self, x_train, y_train, x_val, y_val, lr=0.001, n_epochs=100, verbose=True):
        best_val_acc = 0.0
        for epoch in range(n_epochs):
            train_loss = 0.0
            train_acc = 0.0
            for i in range(len(x_train)):
                x = x_train[i].reshape(-1, 1)
                y = y_train[i]
                q_values = self.forward(x)
                probabilities = self._softmax(q_values / self.temperature)
                loss = -np.log(probabilities[y])
                grad_q_values = probabilities
                grad_q_values[y] -= 1
                grad_q_values /= self.temperature
                grad_W2 = np.dot(grad_q_values, self.model['W2'])
                grad_b2 = grad_q_values
                grad_a1 = np.multiply(grad_W2.T, np.where(q_values > 0, 1, 0))
                grad_W1 = np.dot(grad_a1, x.T)
                grad_b1 = grad_a1
                self.model['W1'] -= lr * grad_W1
                self.model['b1'] -= lr * grad_b1
                self.model['W2'] -= lr * grad_W2.T
                self.model['b2'] -= lr * grad_b2
                train_loss += loss
                train_acc += int(np.argmax(q_values) == y)
            train_loss /= len(x_train)
            train_acc /= len(x_train)

            val_loss = 0.0
            val_acc = 0.0
            for i in range(len(x_val)):
                x = x_val[i].reshape(-1, 1)
                y = y_val[i]
                q_values = self.forward(x)
                probabilities = self._softmax(q_values / self.temperature)
                loss = -np.log(probabilities[y])
                val_loss += loss
                val_acc += int(np.argmax(q_values) == y)
            val_loss /= len(x_val)
            val_acc /= len(x_val)

            if verbose:
                print("Epoch %d, train loss: %.4f, train acc: %.4f, val loss: %.4f, val acc: %.4f" % (
                    epoch+1, train_loss, train_acc, val_loss, val_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_temperature = self.temperature

        self.temperature = best_temperature
        if verbose:
            print("Optimal temperature: %.4f" % best_temperature)
