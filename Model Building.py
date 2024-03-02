import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BiDirectionalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output):
        super(BiDirectionalRNN,self).__init__()

        # BiDirectional LSTM layer

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_size, output)

    def forward(self, x):
        output, _ = self.lstm(x)


        out = torch.cat((output[:, -1, :output.size(2) // 2], output[:, 0, output.size(2) // 2:]), dim=1)

        out = self.fc1(out)

        return out


input_size = 10
hidden_size = 20
output_size = 5

# Creating model

model = BiDirectionalRNN(input_size, hidden_size, output_size)

batch_size = 3
sequence_length = 15
input_sequence = torch.randn(batch_size, sequence_length, input_size).to(device)

output = model(input_sequence).to(device)

print('Input_shape', input_sequence.shape)
print('Output', output.shape)
