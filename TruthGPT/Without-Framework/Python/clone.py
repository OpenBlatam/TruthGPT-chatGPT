import tensorflow as tf

class ChatGPT(tf.keras.Model):
    def __init__(self,vocab_size,embed_dim,hidden_dim,num_layers):
        super().__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size,embed_dim)
        self.lstm = tf.keras.layers.LSTM(hidden_dim,return_sequences=True,stateful=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self,inputs,states):
        x = self.embed(inputs)
        x,states = self.lstm(x,states)
        x = self.dropout(x)
        x = self.dense(x)
        return x,states

    def predict(self,inputs):
        states = self.initial_state()
        for i in range(inputs.shape[1]):
            x,states = self(inputs[:,i],states)
        predictions = x
        return predictions

    def train(self,inputs,targets,epochs=10,batch_size=32):
        optimizer = tf.keras.optimizers.Adam(0.001)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = 0
            for i in range(0,inputs.shape[0],batch_size):
                x,y = inputs[i:i+batch_size],targets[i:i+batch_size]
                x,states = self(x,self.initial_state())
                loss += self.loss(x,y)
            loss.backward()
            optimizer.step()
        print(f"Loss after epoch {epoch}: {loss.numpy()}")

