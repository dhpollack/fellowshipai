import torch
import torch.nn as nn

class CharacterLID(nn.Module):
    def __init__(self, num_emb_chars, num_emb_out=100, num_classes=21):
        super(CharacterLID, self).__init__()
        self.emb = nn.EmbeddingBag(num_emb_chars, num_emb_out)
        self.linear = nn.Linear(num_emb_out, num_classes)
        self.activation = nn.LogSoftmax()
    def forward(self, input):
        x = self.emb(input)
        x = self.linear(x)
        return x
