import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_PATH = "data/chopin_sequences.npz"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "chopin_lstm.pt")

SEQ_LEN = 48       # length of input sequences
BATCH_SIZE = 64
EPOCHS = 10       # keep small so it finishes
LR = 5e-4

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class ChopinDataset(Dataset):
    def __init__(self, pitch_sequence, seq_len):
        # Filter out rests (-1) OPTIONAL: you can keep rests if you want
        self.seq = pitch_sequence.astype(int)

        # Map pitches to a compact index space [0..num_tokens-1]
        unique_tokens = np.unique(self.seq)
        self.token_to_idx = {tok: i for i, tok in enumerate(unique_tokens)}
        self.idx_to_token = {i: tok for tok, i in self.token_to_idx.items()}
        self.vocab_size = len(unique_tokens)

        # Encode full sequence as indices
        self.encoded = np.array([self.token_to_idx[t] for t in self.seq], dtype=int)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.encoded) - self.seq_len

    def __getitem__(self, idx):
        x = self.encoded[idx : idx + self.seq_len]
        y = self.encoded[idx + 1 : idx + self.seq_len + 1]
        # x, y shape: [SEQ_LEN]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class LSTMMusicModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: [B, T]
        x = self.embed(x)  # [B, T, E]
        out, hidden = self.lstm(x, hidden)  # out: [B, T, H]
        logits = self.fc(out)  # [B, T, V]
        return logits, hidden


def load_data():
    data = np.load(DATA_PATH)
    pitch_seq = data["pitch_sequence"]
    print(f"Loaded pitch sequence of length {len(pitch_seq)}")
    ds = ChopinDataset(pitch_seq, SEQ_LEN)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    return ds, dl


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    ds, dl = load_data()
    print(f"Vocab size: {ds.vocab_size}")

    model = LSTMMusicModel(vocab_size=ds.vocab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in dl:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(x)  # logits: [B, T, V]
            # reshape for CE: (B*T, V) vs (B*T)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), y.view(B * T))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dl)
        print(f"Epoch {epoch}/{EPOCHS} - loss: {avg_loss:.4f}")

    # Save model + token maps
    save_dict = {
        "model_state": model.state_dict(),
        "token_to_idx": ds.token_to_idx,
        "idx_to_token": ds.idx_to_token,
        "vocab_size": ds.vocab_size,
        "seq_len": SEQ_LEN,
    }
    torch.save(save_dict, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
