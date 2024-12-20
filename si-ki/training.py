import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# **Dataset-Klasse**
class SignatureDataset(Dataset):
    def __init__(self, data_file, max_len=256):
        with open(data_file, 'r') as f:
            data = json.load(f)
        self.inputs = [d['input'] for d in data]
        self.outputs = [d['output'] for d in data]
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output = self.outputs[idx]

        # Konvertiere Eingabe und Ausgabe zu Unicode-Vektoren
        input_vector = [ord(c) for c in input_text[:self.max_len]] + [0] * (self.max_len - len(input_text))
        output_vector = [
            len(output["name"]), len(output["company"]),
            len(output["website"]), len(output["phone"])
        ]

        return torch.tensor(input_vector, dtype=torch.float32), torch.tensor(output_vector, dtype=torch.float32)

# **Dataset laden**
data_file = 'training_data.json'
dataset = SignatureDataset(data_file)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# **Modell erstellen**
class SignatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SignatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Modell-Parameter
input_size = 256  # Maximale Länge des Textes
hidden_size = 128
output_size = 4  # Name, Company, Website, Phone

# Initialisiere Modell, Optimierer und Verlustfunktion
model = SignatureExtractor(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

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

# Modell speichern
torch.onnx.export(
    model,
    torch.randn(1, input_size),
    "signature_extractor.onnx",
    input_names=["input"],
    output_names=["output"]
)
print("Model successfully saved!")
