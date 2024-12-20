import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# **Dataset-Klasse mit Padding**
class SignatureDataset(Dataset):
    def __init__(self, data_file, max_input_len=256, max_output_len=128):
        with open(data_file, 'r') as f:
            data = json.load(f)
        self.inputs = [d['input'] for d in data]
        self.outputs = [d['output'] for d in data]
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output = self.outputs[idx]

        # Eingabesequenz auf fixe Länge bringen
        input_vector = [ord(c) for c in input_text[:self.max_input_len]] + [0] * (self.max_input_len - len(input_text))

        # Ausgabesequenz erstellen (Name, Firma, Website, Telefon) und mit "|" trennen
        output_text = output["name"] + "|" + output["company"] + "|" + output["website"] + "|" + output["phone"]
        output_vector = [ord(c) for c in output_text[:self.max_output_len]] + [0] * (self.max_output_len - len(output_text))

        return torch.tensor(input_vector, dtype=torch.float32), torch.tensor(output_vector, dtype=torch.float32)

# **Seq2Seq-Modell**
class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = self.decoder(x)
        return x

# **Parameter**
input_size = 256  # Maximale Eingabelänge
hidden_size = 128  # Größe des versteckten Layers
output_size = 128  # Maximale Ausgabelänge

# **Dataset und DataLoader**
data_file = 'training_data.json'  # Trainingsdaten im JSON-Format
dataset = SignatureDataset(data_file, max_input_len=input_size, max_output_len=output_size)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# **Modell, Optimierer und Verlustfunktion initialisieren**
model = Seq2SeqModel(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# **Training**
epochs = 50
for epoch in range(epochs):
    for inputs, outputs in dataloader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, outputs)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# **Modell speichern**
torch.onnx.export(
    model,
    torch.randn(1, input_size),
    "signature_extractor_seq2seq.onnx",
    input_names=["input"],
    output_names=["output"]
)
print("Model successfully saved!")
