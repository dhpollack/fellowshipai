import torch
import torch.nn as nn

class FTModel(nn.Module):
    def __init__(self, input_dim = 100, layer_sizes = [1000,200], num_classes = 21):
        super(FTModel, self).__init__()
        self.norm_in = nn.LayerNorm([input_dim])
        self.act = nn.LeakyReLU()
        #self.act = nn.Sigmoid()
        self.blocks = nn.ModuleList()
        prev_layer = input_dim
        for next_layer in layer_sizes:
            self.blocks.append(self.build_block(prev_layer, next_layer))
            prev_layer = next_layer
        self.classifier = nn.Linear(layer_sizes[-1], num_classes)

    def forward(self, input):
        x = input
        x = self.norm_in(x)
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)

        return x

    def build_block(self, input_dim, output_dim):
        seq = nn.Sequential(*[
            nn.Linear(input_dim, output_dim),
            self.act,
            nn.LayerNorm(output_dim)
        ])
        return seq
