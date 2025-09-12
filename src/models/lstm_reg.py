import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    """Many-to-one LSTM: input [B,T,F] -> predicts next-day close_z scalar."""
    def __init__(self, F, hidden=128, layers=2, dropout=0.2):
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
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: [B,T,F]
        h, _ = self.lstm(x)         # [B,T,H]
        last = h[:, -1, :]          # [B,H]
        y = self.head(last)         # [B,1] (close_z)
        return y.squeeze(-1)        # [B]