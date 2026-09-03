import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    densenet121, DenseNet121_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)
from transformers import AutoModel
import config

class ImageEncoder(nn.Module):
    """
    Modular visual encoder supporting DenseNet-121, ResNet-50, and EfficientNet-B0 backbones.
    """
    def __init__(self, backbone_name="densenet121"):
        super(ImageEncoder, self).__init__()
        self.backbone_name = backbone_name.lower()
        
        if self.backbone_name == "densenet121":
            weights = DenseNet121_Weights.DEFAULT
            self.model = densenet121(weights=weights)
            # Remove final classifier to output visual embeddings
            self.feature_dim = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
            # Reference to the last conv layer for Grad-CAM
            self.last_conv_layer = self.model.features[-1]
            
        elif self.backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT
            self.model = resnet50(weights=weights)
            self.feature_dim = self.model.fc.in_features
            self.model.fc = nn.Identity()
            self.last_conv_layer = self.model.layer4[-1]
            
        elif self.backbone_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT
            self.model = efficientnet_b0(weights=weights)
            self.feature_dim = self.model.classifier[1].in_features
            self.model.classifier = nn.Identity()
            self.last_conv_layer = self.model.features[-1]
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x):
        # Outputs shape: [batch_size, feature_dim]
        features = self.model(x)
        return features

class TextEncoder(nn.Module):
    """
    ClinicalBERT encoder returning dense text representations.
    """
    def __init__(self, model_name=config.TEXT_MODEL_NAME, freeze=config.FREEZE_TEXT_ENCODER):
        super(TextEncoder, self).__init__()
        # Load pre-trained ClinicalBERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.feature_dim = self.bert.config.hidden_size  # Typically 768
        
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Forward pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract representation from the CLS token (index 0)
        # Shape: [batch_size, feature_dim]
        text_embedding = outputs.last_hidden_state[:, 0, :]
        return text_embedding

class LateFusionClassifier(nn.Module):
    """
    Late Fusion architecture combining visual and clinical note embeddings.
    Handles optional text embeddings using projection alignment and zero-fallbacks.
    """
    def __init__(self, image_backbone="densenet121", shared_dim=256):
        super(LateFusionClassifier, self).__init__()
        self.image_encoder = ImageEncoder(backbone_name=image_backbone)
        self.text_encoder = TextEncoder()
        
        # Projections to align visual and textual features into a shared space
        self.img_proj = nn.Linear(self.image_encoder.feature_dim, shared_dim)
        self.txt_proj = nn.Linear(self.text_encoder.feature_dim, shared_dim)
        
        # Classification MLP head taking concatenated representations
        # Shape: shared_dim (visual) + shared_dim (textual) = 2 * shared_dim
        self.classifier = nn.Sequential(
            nn.Linear(2 * shared_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(config.PATHOLOGIES))
        )
        
    def forward(self, image, input_ids, attention_mask, text_available):
        # 1. Get image and text embeddings
        img_emb = self.image_encoder(image)
        txt_emb = self.text_encoder(input_ids, attention_mask)
        
        # 2. Project features to shared subspace
        proj_img = F.relu(self.img_proj(img_emb))
        proj_txt = F.relu(self.txt_proj(txt_emb))
        
        # 3. Apply missing-text fallback: zero out text embedding if unavailable
        # text_available shape: [batch_size], expand to [batch_size, 1] for multiplication
        proj_txt = proj_txt * text_available.unsqueeze(-1)
        
        # 4. Concatenate and classify
        fused = torch.cat([proj_img, proj_txt], dim=-1)
        logits = self.classifier(fused)
        
        return logits

class ImageOnlyClassifier(nn.Module):
    """
    Image-only baseline classifier wrapping visual encoder and linear classification head.
    """
    def __init__(self, backbone_name="densenet121"):
        super(ImageOnlyClassifier, self).__init__()
        self.image_encoder = ImageEncoder(backbone_name=backbone_name)
        self.classifier = nn.Linear(self.image_encoder.feature_dim, len(config.PATHOLOGIES))

    def forward(self, x):
        features = self.image_encoder(x)
        logits = self.classifier(features)
        return logits

if __name__ == "__main__":
    # Quick dimensionality test
    print("Testing model dimensions...")
    img = torch.randn(2, 3, 224, 224)
    input_ids = torch.zeros(2, config.MAX_TEXT_LENGTH, dtype=torch.long)
    attention_mask = torch.zeros(2, config.MAX_TEXT_LENGTH, dtype=torch.long)
    text_available = torch.tensor([1.0, 0.0], dtype=torch.float32)
    
    # Visual Baseline
    img_model = ImageOnlyClassifier("densenet121")
    out_img = img_model(img)
    print("Visual baseline output shape:", out_img.shape)
    assert out_img.shape == (2, 14)
    
    # Multimodal late fusion
    fusion_model = LateFusionClassifier("densenet121")
    out_fusion = fusion_model(img, input_ids, attention_mask, text_available)
    print("Multimodal fusion output shape:", out_fusion.shape)
    assert out_fusion.shape == (2, 14)
    print("Tests completed successfully!")
