import json
import torch
import torch.nn as nn
import torch.optim as optim
import re

# **Daten laden**
def load_training_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Vorverarbeitung: Extraktion und Vektorisierung
def preprocess_data(data):
    inputs, outputs = [], []
    for entry in data:
        signature = entry["signature"]
        inputs.append(signature)

        # Kombinieren der Zielwerte zu einem stringbasierten Format
        outputs.append({
            "name": entry["name"],
            "company": entry["company"],
            "website": entry["website"],
            "phone": entry["phone"]
        })
    return inputs, outputs

# Textvektorisierung (dummy): In realen Szenarien kannst du einen Text-Encoder verwenden
def vectorize_text(text, max_len=256):
    vector = [ord(c) for c in text[:max_len]]  # Konvertiere Zeichen zu Unicode-Werten
    return vector + [0] * (max_len - len(vector))  # Padding

# Trainingsdaten laden
data_file = 'training_data.json'
training_data = load_training_data(data_file)
inputs, outputs = preprocess_data(training_data)

# Vektorisieren
max_len = 256  # Maximale Länge der Signatur
input_vectors = [vectorize_text(text, max_len) for text in inputs]

# Dummy-Ausgabevektoren (z. B. Indexnummern von Kategorien, in echten Modellen komplexer)
output_vectors = [
    [
        len(entry["name"]),  # Länge des Namens als Dummy-Ausgabe
        len(entry["company"]),  # Länge der Firma
        len(entry["website"]),  # Länge der Website
        len(entry["phone"])  # Länge der Telefonnummer
    ]
    for entry in outputs
]

# Tensoren erstellen
input_tensors = torch.tensor(input_vectors, dtype=torch.float32)
output_tensors = torch.tensor(output_vectors, dtype=torch.float32)

# Modell erstellen
class SignatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SignatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Parameter
input_size = max_len
hidden_size = 128
output_size = 4  # Name, Firma, Website, Telefonnummer

# Modell initialisieren
model = SignatureExtractor(input_size, hidden_size, output_size)

# Optimierer und Verlustfunktion
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(input_tensors)
    loss = criterion(predictions, output_tensors)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Modell exportieren
dummy_input = torch.randn(1, max_len)
torch.onnx.export(
    model,
    dummy_input,
    "signature_extractor.onnx",
    input_names=["input"],
    output_names=["output"]
)

print("Modell erfolgreich exportiert als 'signature_extractor.onnx'")