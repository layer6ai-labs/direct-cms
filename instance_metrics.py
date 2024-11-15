import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import clip


class AestheticsMLP(pl.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


class InstanceMetrics():
    def __init__(self, device):
        self.device = device

        # Load CLIP model
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=device)

        # Load aesthetics predictor MLP head
        self.aesthetics_mlp = AestheticsMLP(768)  # CLIP embedding dim is 768 for CLIP ViT-L/14
        s = torch.load("data/aesthetics_mlp_weights/sac+logos+ava1-l14-linearMSE.pth")
        self.aesthetics_mlp.load_state_dict(s)
        self.aesthetics_mlp.to(device)
        self.aesthetics_mlp.eval()

    @torch.inference_mode()    
    def compute_instance_metrics(self, pil_image, text):
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        text_tok = clip.tokenize([text], truncate=True).to(self.device)

        image_features = self.clip_model.encode_image(image).float()
        text_features = self.clip_model.encode_text(text_tok).float()
        image_features_norm = F.normalize(image_features)
        text_features_norm = F.normalize(text_features)

        clip_score = (image_features_norm * text_features_norm).sum().item()
        aesthetic_score = self.aesthetics_mlp(image_features_norm).item()

        return clip_score, aesthetic_score