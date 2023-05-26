import math
import random
import numpy as np

class NonAutoregressiveTransformer:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()

    def _create_encoder(self):
        encoder = []
        for i in range(6):
            encoder.append(self._create_encoder_layer())

        return nn.Sequential(*encoder)

    def _create_encoder_layer(self):
        return nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1
        )

    def _create_decoder(self):
        decoder = []
        for i in range(6):
            decoder.append(self._create_decoder_layer())

        return nn.Sequential(*decoder)

    def _create_decoder_layer(self):
        return nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1
        )

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids, attention_mask)
        decoder_outputs = self.decoder(
            input_ids=None,
            attention_mask=attention_mask,
            memory=encoder_outputs
        )

        logits = self.linear(decoder_outputs[0])

        return logits

    def generate(self, starting_token):
        batch_size = 1
        seq_len = 1

        # Convert the starting token to a tensor.
        starting_token_tensor = torch.tensor([starting_token], dtype=torch.long)

        # Generate the next token.
        for i in range(seq_len):
            # Get the output of the decoder.
            decoder_outputs = self.decoder(
                input_ids=starting_token_tensor,
                attention_mask=None,
                memory=None
            )

            # Get the logits for the next token.
            logits = self.linear(decoder_outputs[0])

            # Sample the next token.
            next_token_id = torch.argmax(logits, dim=-1).item()

            # Update the starting token.
            starting_token_tensor = torch.tensor([next_token_id], dtype=torch.long)

        # Convert the generated tokens to a string.
        generated_tokens = starting_token_tensor.tolist()

        return generated_tokens

