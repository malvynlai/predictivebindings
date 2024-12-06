import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from train import MoleculePredictor, MoleculeDataset  


def predict(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            predictions.append(probs.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    return predictions


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_path = '/home/cs229/leash-BELKA/test_enc.parquet'
    test_dataset = MoleculeDataset(test_path, test=True)    
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        num_workers=4,
        pin_memory=True
    )
    model = MoleculePredictor(test=True)
    model.load_state_dict(torch.load('/home/cs229/ckpt/model_1.pt'))
    model = model.to(device)
    predictions = predict(model, test_loader, device)
    final = pd.read_parquet('/home/cs229/leash-BELKA/test.parquet')
    final['binds'] = 0
    final.loc[final['protein_name']=='BRD4', 'binds'] = predictions[(final['protein_name']=='BRD4').values, 0]
    final.loc[final['protein_name']=='HSA', 'binds'] = predictions[(final['protein_name']=='HSA').values, 1]
    final.loc[final['protein_name']=='sEH', 'binds'] = predictions[(final['protein_name']=='sEH').values, 2]
    final[['id', 'binds']].to_csv('submission.csv', index = False)
    print("Predictions saved to submission.csv")
    

if __name__ == '__main__':
    main() 