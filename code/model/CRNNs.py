import os
import torch
import torchaudio
import torch.nn as nn
from tqdm import tqdm
import random
random.seed(1984)
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class CRNN10(nn.Module):
    def __init__(self, up_model, up_out_dim):
        super(CRNN10, self).__init__()
        self.upstream_model = up_model
        self.upstream_feat_dim = up_out_dim

        # Add 4 ConvBlocks
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=up_out_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Existing MLP layers
        self.overall_mlp_layer1 = nn.Linear(in_features=1024, out_features=256)
        self.overall_mlp_layer2 = nn.Linear(in_features=256, out_features=64)
        self.overall_mlp_layer3 = nn.Linear(in_features=64, out_features=1)
        self.textual_mlp_layer1 = nn.Linear(in_features=self.upstream_feat_dim * 2, out_features=256)
        self.textual_mlp_layer1 = nn.Linear(in_features=1536, out_features=256)
        self.textual_mlp_layer2 = nn.Linear(in_features=256, out_features=64)
        self.textual_mlp_layer3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, wavs, texts):
        wav_embed = self.upstream_model.get_audio_embedding_from_data(wavs, use_tensor=True).to(device)
        text_embed = self.upstream_model.get_text_embedding(texts, use_tensor=True).to(device)

        # Expand sequence length
        wav_embed = wav_embed.unsqueeze(-1)  # Add a sequence length dimension
        wav_embed = wav_embed.repeat(1, 1, 16)  # Repeat along the sequence length dimension (e.g., 10 times)

        # Pass through ConvBlocks
        # print(wav_embed.shape)
        wav_embed = self.conv_block1(wav_embed)
        wav_embed = self.conv_block2(wav_embed)
        wav_embed = self.conv_block3(wav_embed)
        wav_embed = self.conv_block4(wav_embed)
        wav_embed = wav_embed.mean(dim=2)  # Global average pooling over the sequence length

        combine_embed = torch.cat((wav_embed, text_embed), dim=1)  # bs*1024        

        hidden1 = self.overall_mlp_layer1(wav_embed)
        hidden1_2 = self.overall_mlp_layer2(hidden1)
        out1 = self.overall_mlp_layer3(hidden1_2)
        hidden2 = self.textual_mlp_layer1(combine_embed)
        hidden2_2 = self.textual_mlp_layer2(hidden2)
        out2 = self.textual_mlp_layer3(hidden2_2)
        return out1, out2
    

class CRNN10_v2(nn.Module):
    def __init__(self, up_model, up_out_dim):
        super(CRNN10_v2, self).__init__()
        self.upstream_model = up_model
        self.upstream_feat_dim = up_out_dim

        # Add 6 ConvBlocks (2 additional blocks added)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=up_out_dim, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Update MLP layers to match the new output dimensions
        self.overall_mlp_layer1 = nn.Linear(in_features=2048, out_features=256)
        self.overall_mlp_layer2 = nn.Linear(in_features=256, out_features=64)
        self.overall_mlp_layer3 = nn.Linear(in_features=64, out_features=1)
        self.textual_mlp_layer1 = nn.Linear(in_features=self.upstream_feat_dim * 2, out_features=256)
        self.textual_mlp_layer2 = nn.Linear(in_features=256, out_features=64)
        self.textual_mlp_layer3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, wavs, texts):
        wav_embed = self.upstream_model.get_audio_embedding_from_data(wavs, use_tensor=True).to(device)
        text_embed = self.upstream_model.get_text_embedding(texts, use_tensor=True).to(device)

        # Expand sequence length
        wav_embed = wav_embed.unsqueeze(-1)  # Add a sequence length dimension
        wav_embed = wav_embed.repeat(1, 1, 16)  # Repeat along the sequence length dimension 

        # Pass through ConvBlocks
        wav_embed = self.conv_block1(wav_embed)
        wav_embed = self.conv_block2(wav_embed)
        wav_embed = self.conv_block3(wav_embed)
        wav_embed = self.conv_block4(wav_embed)
        wav_embed = wav_embed.mean(dim=2)  # Global average pooling over the sequence length

        # print("=====", (wav_embed.shape, "=====", text_embed.shape))
        combine_embed = torch.cat((wav_embed, text_embed), dim=1)  # bs*2048        

        hidden1 = self.overall_mlp_layer1(wav_embed)
        hidden1_2 = self.overall_mlp_layer2(hidden1)
        out1 = self.overall_mlp_layer3(hidden1_2)
        hidden2 = self.textual_mlp_layer1(combine_embed)
        hidden2_2 = self.textual_mlp_layer2(hidden2)
        out2 = self.textual_mlp_layer3(hidden2_2)
        return out1, out2