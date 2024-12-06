import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from train import MoleculeDataset
from torch.utils.data import DataLoader
from model import CNN1d, CNN1dWithGRU, CNN1dWithoutGRU


def load_models(model_dir='trained_models'):
    models = {}
    model_dir = Path(model_dir)
    for model_path in model_dir.glob('*.pt'):
        models[model_path.stem] = torch.load(model_path, map_location='cpu')
    return models


def plot_weight_distributions(models, save_path='weight_distributions.png'):
    # Distribution of weights for each model
    plt.figure(figsize=(15, 5))
    for idx, (name, state_dict) in enumerate(models.items(), 1):
        weights = []
        for param_name, param in state_dict.items():
            if 'weight' in param_name:
                weights.extend(param.cpu().numpy().flatten())
        plt.subplot(1, 3, idx)
        sns.histplot(weights, bins=50, kde=True)
        plt.title(f'{name}\nWeight Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_layer_sparsity(models, save_path='layer_sparsity.png'):
    # Sparsity in each layer
    plt.figure(figsize=(12, 6))
    
    for name, state_dict in models.items():
        sparsities = []
        layer_names = []
        
        for param_name, param in state_dict.items():
            if 'weight' in param_name:
                sparsity = (param.cpu().abs() < 1e-6).float().mean().item() * 100
                sparsities.append(sparsity)
                layer_names.append(param_name.split('.')[0])
        
        plt.plot(sparsities, 'o-', label=name)
        plt.xticks(range(len(layer_names)), layer_names, rotation=45)
    
    plt.title('Layer-wise Weight Sparsity')
    plt.xlabel('Layer')
    plt.ylabel('Sparsity (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_attention_patterns(models, save_path='attention_patterns.png'):
    # Attention patterns for models with attention mechanisms 
    plt.figure(figsize=(15, 5))
    
    for idx, (name, state_dict) in enumerate(models.items(), 1):
        attn_weights = None
        for param_name, param in state_dict.items():
            if 'attention' in param_name.lower() and 'weight' in param_name:
                attn_weights = param.cpu()
                break
        
        if attn_weights is not None:
            plt.subplot(1, 3, idx)
            sns.heatmap(attn_weights.numpy(), cmap='viridis')
            plt.title(f'{name}\nAttention Weights')
            plt.xlabel('Query dimension')
            plt.ylabel('Key dimension')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_parameter_magnitudes(models, save_path='parameter_magnitudes.png'):
    # Magnitude of parameters across different layers
    plt.figure(figsize=(12, 6))
    
    for name, state_dict in models.items():
        magnitudes = []
        layer_names = []
        
        for param_name, param in state_dict.items():
            if 'weight' in param_name:
                magnitude = torch.norm(param).item()
                magnitudes.append(magnitude)
                layer_names.append(param_name.split('.')[0])
        
        plt.plot(magnitudes, 'o-', label=name)
        plt.xticks(range(len(layer_names)), layer_names, rotation=45)
    
    plt.title('Layer-wise Parameter Magnitudes')
    plt.xlabel('Layer')
    plt.ylabel('Frobenius Norm')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_activation_histograms(models, dataset, save_path='activation_histograms.png'):
    plt.figure(figsize=(15, 5))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    batch = next(iter(loader))
    input_data = batch['features']
    model_classes = {
        'cnn1d': CNN1d,
        'cnn1d_with_gru': CNN1dWithGRU,
        'cnn1d_without_gru': CNN1dWithoutGRU
    }
    
    for idx, (name, state_dict) in enumerate(models.items(), 1):
        if name not in model_classes:
            print(f"Warning: No matching model class for {name}")
            continue
        model_class = model_classes[name]
        if name == 'cnn1d':
            model = model_class(vocab_size=37, hidden_dim=64)
        else:
            model = model_class(vocab_size=37, embedding_dim=64)
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            activations = []
            def hook(module, input, output):
                activations.append(output.cpu().numpy())
            handles = []
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    handles.append(module.register_forward_hook(hook))
            _ = model(input_data)
            for handle in handles:
                handle.remove()
        
        plt.subplot(1, 3, idx)
        for i, act in enumerate(activations):
            sns.histplot(act.flatten(), bins=50, alpha=0.5, label=f'Layer {i+1}')
        plt.title(f'{name}\nActivation Distribution')
        plt.xlabel('Activation Value')
        plt.ylabel('Count')
        if len(activations) <= 5:  
            plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    models = load_models()
    dataset = MoleculeDataset('/home/cs229/subset/enc_train.parquet')
    
    plot_weight_distributions(models)
    plot_layer_sparsity(models)
    plot_attention_patterns(models)
    plot_parameter_magnitudes(models)
    plot_activation_histograms(models, dataset)


if __name__ == '__main__':
    main()
