import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Prediction:
    def __init__(self, model, feature_scaler, label_scaler, config):
        self.model = model
        self.feature_scaler = feature_scaler
        self.label_scaler = label_scaler
        self.device = torch.device(config['device'])

    def predict(self, sequence):
        sequence = torch.tensor(sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            try:
                elbow_prediction,wrist_prediction,_ = self.model(sequence)
                prediction = torch.cat((elbow_prediction,wrist_prediction), dim=1)
                return self.label_scaler.inverse_transform(prediction.cpu().detach().numpy())
            except Exception as e:
                logger.error("Prediction error: %s", e)
                return None