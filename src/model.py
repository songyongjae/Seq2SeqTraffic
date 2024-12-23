import torch
import torch.nn as nn
import random


class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(lstm_encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x_input):
        lstm_out, hidden = self.lstm(x_input)
        return lstm_out, hidden


class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(lstm_decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, hidden = self.lstm(x_input.unsqueeze(-1), encoder_hidden_states)
        output = self.linear(lstm_out)
        return output, hidden


class lstm_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstm_encoder_decoder, self).__init__()
        self.encoder = lstm_encoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = lstm_decoder(input_size=input_size, hidden_size=hidden_size)

    def forward(self, inputs, targets, target_len, teacher_forcing_ratio):
        batch_size, input_size = inputs.shape[0], inputs.shape[2]
        outputs = torch.zeros(batch_size, target_len, input_size)
        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:, -1, :]

        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden)
            out = out.squeeze(1)
            if random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, t, :]
            else:
                decoder_input = out
            outputs[:, t, :] = out

        return outputs

    def predict(self, inputs, target_len):
        self.eval()
        inputs = inputs.unsqueeze(0)
        outputs = torch.zeros(inputs.shape[0], target_len, inputs.shape[2])
        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:, -1, :]

        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden)
            out = out.squeeze(1)
            decoder_input = out
            outputs[:, t, :] = out

        return outputs.detach().numpy()[0, :, 0]
