import torch.nn as nn

from timm.models.layers import trunc_normal_
from Backbone import load_DINO, Decoder, Gen_2D_embedding_from_vector


class Face_prompt(nn.Module):
    def __init__(self, model_type):
        super(Face_prompt, self).__init__()
        self.width = 384
        self.nhead = 6

        self.Linear = nn.Sequential(nn.Linear(self.width, 256),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(256, 2))

        self.apply(self._init_weights)
        if model_type == 'small':
            self.Encoder = load_DINO("small", 16)
        elif model_type == 'base':
            self.Encoder = load_DINO("small", 8)
        else:
            raise NotImplementedError

        self.query_gen = Gen_2D_embedding_from_vector(self.width)
        self.Decoder = Decoder(self.width, self.nhead, 6, self.width * 4)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x, meanshape):
        meanshape = meanshape * 2.0 - 1.0

        landmark_query = self.query_gen(meanshape)

        Feature, Feature_pos = self.Encoder(x)
        landmark_feature = self.Decoder(Feature, Feature_pos, landmark_query.to(Feature.device))

        out = self.Linear(landmark_feature)

        return out
