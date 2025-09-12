import torch
import torch.nn as nn

class GeneratorLSTM(nn.Module):
    """
    LSTM Generator:
      input  : [B, T, F] (history of T days with F features)
      output : [B, F]    (next-day feature vector in z-space; we'll use [:, close_idx])
    """
    def __init__(self, F, hidden=128, layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=F,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden, F),
        )

    def forward(self, x):
        h, _ = self.lstm(x)      # [B,T,H]
        last = h[:, -1, :]       # [B,H]
        y = self.head(last)      # [B,F]
        return y


class DiscriminatorMLP(nn.Module):
    """
    MLP Discriminator:
      input  : concat([hist.flatten(), next_vec]) = T*F + F
      output : scalar probability real/fake
    """
    def __init__(self, T, F, hidden=(72, 100, 10), dropout=0.2):
        super().__init__()
        D_in = T * F + F
        layers = []
        prev = D_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, hist, nxt):
        b = hist.size(0)
        x = torch.cat([hist.reshape(b, -1), nxt], dim=1)
        return self.net(x).squeeze(-1)  # [B]