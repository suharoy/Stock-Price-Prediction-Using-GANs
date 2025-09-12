import torch
import torch.nn as nn

class GeneratorLSTMCond(nn.Module):
    """
    Conditional generator:
      input  : hist [B,T,F], cond [B,C]
      output : next-day vector [B,F]
      We concat cond to the last hidden state before the head.
    """
    def __init__(self, F, C, hidden=128, layers=1, dropout=0.2):
        super().__init__()
        self.F, self.C = F, C
        self.lstm = nn.LSTM(
            input_size=F,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + C, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, F),
        )

    def forward(self, hist, cond):
        # hist: [B,T,F], cond: [B,C]
        h, _ = self.lstm(hist)      # [B,T,H]
        last = h[:, -1, :]          # [B,H]
        x = torch.cat([last, cond], dim=1)
        out = self.head(x)          # [B,F]
        return out


class DiscriminatorMLPCond(nn.Module):
    """
    Conditional discriminator:
      input : concat(hist.flatten, next_vec, cond) -> prob real/fake
    """
    def __init__(self, T, F, C, hidden=(72,100,10), dropout=0.2):
        super().__init__()
        D_in = T*F + F + C
        layers = []
        prev = D_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, hist, nxt, cond):
        b = hist.size(0)
        x = torch.cat([hist.reshape(b, -1), nxt, cond], dim=1)
        return self.net(x).squeeze(-1)