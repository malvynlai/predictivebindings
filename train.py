import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from model import MoleculePredictor, ModelWithGRU, ModelWithoutGRU
import matplotlib.pyplot as plt


class MoleculeDataset(Dataset):
    def __init__(self, parquet_file, test=False):
        print(f"\nLoading data from: {parquet_file}")
        
        if not os.path.exists(parquet_file):
            raise FileNotFoundError(f"File not found: {parquet_file}")

        data = pd.read_parquet(parquet_file)
        self.ids = data['molecule_id'].values if 'molecule_id' in data.columns else None
 
        try:
            feature_cols = [f'enc{i}' for i in range(142)]
            missing_cols = [col for col in feature_cols if col not in data.columns]

            if missing_cols:
                raise KeyError(f"Missing columns in dataset: {missing_cols[:5]}...")

            # Load features directly without parallel processing
            self.features = data[feature_cols].values.astype(np.int64)
            
            if not test:
                self.targets = data[['bind1', 'bind2', 'bind3']].values.astype(np.float32)
            else:
                self.targets = None

            print('\nSuccessfully loaded:')
            print(f'Features shape: {self.features.shape}')
            if not test:
                print(f"Targets shape: {self.targets.shape}")

        except KeyError as e:
            print('\nError accessing columns.')
            print(f'Looking for feature columns like: {feature_cols[:5]}')
            if not test:
                print(f"Looking for target columns: ['bind1', 'bind2', 'bind3']")
            print(f"Found columns: {data.columns.tolist()}")
            raise e
        
        self.is_test = test
    

    def __len__(self):
        return len(self.features)
    

    def __getitem__(self, idx):
        if self.is_test:
            return {'features': torch.from_numpy(self.features[idx])}
        return {
            'features': torch.from_numpy(self.features[idx]),
            'targets': torch.from_numpy(self.targets[idx])
        }


def train_loop(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    
    model = model.to(device)
    # Increase batch size since V100 has more memory
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    initial_lr = 2e-3  # Can use slightly higher learning rate with V100
    end_lr = 1e-7
    total_steps = len(train_loader) * num_epochs
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=0.01  
    )
    
    lambda_fn = lambda epoch: (1 - epoch/total_steps)**0.5
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)
    best_val_loss = float('inf')
    best_model = None

    print(f"Training on {device}")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            features = batch['features'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(features)
                loss = criterion(outputs, targets)
            optimizer.zero_grad(set_to_none=True)  
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device, non_blocking=True)
                targets = batch['targets'].to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()  # Save state dict instead of model

        epoch_train_loss = train_loss
        train_losses.append(epoch_train_loss)
        epoch_val_loss = val_loss
        val_losses.append(epoch_val_loss)
        
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()
    
    model.load_state_dict(best_model)  # Load best model before returning
    return model


def train_single_model(model_class, train_dataset, val_dataset, gpu_id, batch_size=2048, epochs=20):
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)  # Explicitly set GPU
    
    model = model_class(vocab_size=37).to(device)
    
    # Increase num_workers for V100 instance which has more CPU cores
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,  # Can use larger batch size for validation
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3
    )
    
    trained_model = train_loop(model, train_loader, val_loader, epochs=epochs, device=device)
    return trained_model

def main():
    # GPU Setup and Monitoring
    num_gpus = torch.cuda.device_count()
    print(f"\nNumber of available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory Available: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    # Increased batch size for V100s
    BATCH_SIZE = 2048  # V100 has 16GB memory
    EPOCHS = 20

    train_path = '/home/cs229/subset/enc_train.parquet'
    val_path = '/home/cs229/val/enc_train9.parquet'
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = MoleculeDataset(train_path)
    val_dataset = MoleculeDataset(val_path)
    
    models_to_train = [
        (MoleculePredictor, 'molecule_predictor'),
        (ModelWithGRU, 'model_with_gru'),
        (ModelWithoutGRU, 'model_without_gru')
    ]
    
    print(f"\nStarting parallel training on {num_gpus} GPUs")
    
    # Train models in parallel using different GPUs
    trained_models = Parallel(n_jobs=len(models_to_train), backend='threading')(
        delayed(train_single_model)(
            model_class,
            train_dataset,
            val_dataset,
            i % num_gpus,  # Distribute across GPUs
            BATCH_SIZE,
            EPOCHS
        ) for i, (model_class, _) in enumerate(models_to_train)
    )
    
    # Save models
    save_dir = 'trained_models'
    os.makedirs(save_dir, exist_ok=True)
    
    for (_, model_name), trained_model in zip(models_to_train, trained_models):
        save_path = os.path.join(save_dir, f'{model_name}.pt')
        torch.save(trained_model.state_dict(), save_path)
        print(f'Saved model: {save_path}')

if __name__ == '__main__':
    main()
