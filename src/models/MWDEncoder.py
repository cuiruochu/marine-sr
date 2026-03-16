from .common import nn


class Encoder(nn.Module):
    def __init__(self, in_ch, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, hidden_dim, out_ch):
        super(Decoder, self).__init__()
        self.decoder = nn.Conv2d(hidden_dim, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        return self.decoder(x)
