
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from io import StringIO
from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
import os

# 1. Helper: Generate XRD Pattern
def get_xrd_pattern(cif_string, bin_width=0.5):
    try:
        parser = CifParser(StringIO(cif_string))
        structure = parser.get_structures()[0]
        sga = SpacegroupAnalyzer(structure)
        conv_struct = sga.get_conventional_standard_structure()
        calc = XRDCalculator("CuKa")
        pattern = calc.get_pattern(conv_struct)
        
        bins = np.arange(0, 90 + bin_width, bin_width)
        digitized = np.digitize(pattern.x, bins)
        df_pat = pd.DataFrame({'bin': digitized, 'intensity': pattern.y})
        all_bins = pd.DataFrame({'bin': range(1, len(bins))})
        return all_bins.merge(df_pat, on='bin', how='left').groupby('bin')['intensity'].mean().fillna(0).tolist()
    except:
        return [0.0] * int(90/bin_width)

# 2. Autoencoder Model
class XRDAutoencoder(nn.Module):
    def __init__(self, input_dim=180, encoding_dim=30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 60), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(60, encoding_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 60), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(60, input_dim), nn.Sigmoid()
        )
    def forward(self, x):
        enc = self.encoder(x)
        return enc, self.decoder(enc)

# 3. Train & Encode Function (ตัวสำคัญที่ต้องมี)
def train_and_encode(df_raw_xrd, encoding_dim=30, epochs=200):
    print(f"   ...Training Autoencoder ({df_raw_xrd.shape[1]}->{encoding_dim})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_x = torch.tensor(df_raw_xrd.values, dtype=torch.float32).to(device)
    model = XRDAutoencoder(input_dim=df_raw_xrd.shape[1], encoding_dim=encoding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        encoded, decoded = model(tensor_x)
        loss = criterion(decoded, tensor_x)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        encoded_features, _ = model(tensor_x)
        
    return pd.DataFrame(encoded_features.cpu().numpy(), columns=[f"Enc_{i}" for i in range(encoding_dim)])

# 4. Full Pipeline Wrapper
def create_df_xrd_final(raw_data, cif_col='cif', target_col='energy_per_atom'):
    print("--- 🚀 Starting XRD Pipeline ---")
    print("1️⃣ Generating Raw XRD...")
    xrd_list = raw_data[cif_col].apply(lambda x: get_xrd_pattern(x)).tolist()
    df_raw = pd.DataFrame(xrd_list, columns=[f'raw_{i}' for i in range(len(xrd_list[0]))])
    
    print("2️⃣ Encoding...")
    df_encoded = train_and_encode(df_raw, encoding_dim=30, epochs=200)
    
    print("3️⃣ PCA Reduction...")
    pca = PCA(n_components=0.99)
    x_pca = pca.fit_transform(StandardScaler().fit_transform(df_encoded))
    df_pca = pd.DataFrame(x_pca, columns=[f'XRD_PC{i+1}' for i in range(x_pca.shape[1])])
    
    if target_col in raw_data.columns:
        return pd.concat([raw_data[[target_col]].reset_index(drop=True), df_pca], axis=1)
    return df_pca
