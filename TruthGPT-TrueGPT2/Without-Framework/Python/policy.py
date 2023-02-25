import torch
import torch.nn as nn
import torch.optim as optim

class Policy(nn.Module):
    def __init__(self, input_size, output_size, temperature=1.0):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.temperature = temperature
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        q_values = self.model(x)
        return q_values

    def select_action(self, x):
        q_values = self.forward(x)
        probabilities = nn.functional.softmax(q_values / self.temperature, dim=1)
        action = torch.multinomial(probabilities, num_samples=1)
        return action

    def optimize_temperature(self, x_train, y_train, x_val, y_val, lr=0.001, n_epochs=100, verbose=True):
        optimizer = optim.Adam([self.temperature], lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        for epoch in range(n_epochs):
            train_loss = 0.0
            train_acc = 0.0
            self.train()
            for i in range(len(x_train)):
                optimizer.zero_grad()
                x = x_train[i].unsqueeze(0)
                y = y_train[i].unsqueeze(0)
                q_values = self.forward(x)
                loss = loss_fn(q_values / self.temperature, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc += (torch.argmax(q_values, dim=1) == y).sum().item()
            train_loss /= len(x_train)
            train_acc /= len(x_train)

            val_loss = 0.0
            val_acc = 0.0
            self.eval()
            with torch.no_grad():
                for i in range(len(x_val)):
                    x = x_val[i].unsqueeze(0)
                    y = y_val[i].unsqueeze(0)
                    q_values = self.forward(x)
                    loss = loss_fn(q_values / self.temperature, y)
                    val_loss += loss.item()
                    val_acc += (torch.argmax(q_values, dim=1) == y).sum().item()
                val_loss /= len(x_val)
                val_acc /= len(x_val)

            if verbose:
                print("Epoch %d, train loss: %.4f, train acc: %.4f, val loss: %.4f, val acc: %.4f" % (epoch+1, train_loss, train_acc, val_loss, val_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_temperature = self.temperature.item()

        self.temperature = best_temperature
        if verbose:
            print("Optimal temperature: %.4f" % (best_temperature))
