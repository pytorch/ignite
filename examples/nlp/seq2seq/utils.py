import torch

SOS_token = 0
EOS_token = 1

MAX_LENGTH = 90

teacher_forcing_ratio = 0.5

hidden_size = 300
total_loss = 0
learning_rate = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
