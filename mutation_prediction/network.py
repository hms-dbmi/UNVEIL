from torch import Tensor
import torch
import torch.nn as nn
import torchvision.models as models
import random
import loralib as lora


class DropoutOrIdentity(nn.Module):
    def __init__(self, p):
        super(DropoutOrIdentity, self).__init__()
        if p is None or p == 0:
            self.layer = nn.Identity()
        else:
            self.layer = nn.Dropout(p)
    def forward(self, x):
        return self.layer(x)
class MILAttention(nn.Module):
    def __init__(self, featureLength = 768, featureInside = 256,dropout=None):
        '''
        Parameters:
            featureLength: Length of feature passed in from feature extractor(encoder)
            featureInside: Length of feature in MIL linear
            dropout: dropout rate. If None, no dropout
        Output: tensor
            weight of the features
        '''
        super(MILAttention, self).__init__()
        self.featureLength = featureLength
        self.featureInside = featureInside

        self.attention_V = nn.Sequential(
            nn.Linear(self.featureLength, self.featureInside, bias=True),
            nn.Tanh(),
            DropoutOrIdentity(dropout)
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.featureLength, self.featureInside, bias=True),
            nn.Sigmoid(),
            DropoutOrIdentity(dropout)
        )
        self.attention_weights = nn.Linear(self.featureInside, 1, bias=True)
        self.softmax_0 = nn.Softmax(dim=0)
        self.softmax_1 = nn.Softmax(dim=1)

    def forward(self, x: Tensor, nonpad = None) -> Tensor:
        bz, pz, fz = x.shape if len(x.shape) == 3 else (1, *x.shape)
        # x = x.view(bz*pz, fz)
        att_v = self.attention_V(x)
        # print("att_v", att_v.shape)
        att_u = self.attention_U(x)
        # print("att_u", att_u.shape)
        att_v = att_v.view(bz * pz, -1)
        att_u = att_u.view(bz * pz, -1)

        att = self.attention_weights(att_u * att_v)
        # print("att", att.shape)
        weight = att.view(bz, pz, 1)

        if nonpad is not None:
            for idx, i in enumerate(weight):
                weight[idx][:nonpad[idx]] = self.softmax_0(weight[idx][:nonpad[idx]])
                weight[idx][nonpad[idx]:] = 0
        else:
            weight = self.softmax_1(att)
        weight = weight.view(bz, 1, pz)

        return weight


class MILNet(nn.Module):
    def __init__(self, featureLength = 768, linearLength = 256, dropout=None):
        '''
        Parameters:
            featureLength: Length of feature from resnet18
            linearLength:  Length of feature for MIL attention
        Forward:
            weight sum of the features
        '''
        super(MILNet, self).__init__()
        flatten = nn.Flatten(start_dim = 1)
        fc = nn.Linear(512, featureLength, bias=True)

        self.attentionlayer = MILAttention(featureLength, linearLength,dropout=dropout)

    def forward(self, x, lengths):
        if len(x.shape) == 2:
            batch_size = 1
            num_patches, feature_dim = x.shape
        else:
            batch_size, num_patches, feature_dim = x.shape

        weight = self.attentionlayer(x, lengths)
        # print(weight)
        x = x.view(batch_size * num_patches, -1)
        weight = weight.view(batch_size * num_patches, 1)
        x = weight * x
        x = x.view(batch_size, num_patches, -1)
        x = torch.sum(x, dim=1)

        return x

class ClfNet(nn.Module):
    def __init__(self, featureLength=768, 
                 latent_dims= [256, 128],
                 classes=2, dropout=None):
        super(ClfNet, self).__init__()

        self.featureExtractor = MILNet(featureLength=featureLength,dropout=None)
        self.featureLength = featureLength
        # self.fc_target = nn.Sequential(
        #     nn.Linear(featureLength, 256, bias=True),
        #     DropoutOrIdentity(dropout),
        #     nn.ReLU(),
        #     nn.LayerNorm(256),
        #     nn.Linear(256, 128, bias=True),
        #     DropoutOrIdentity(dropout),
        #     nn.ReLU(),
        #     nn.LayerNorm(128),
        #     nn.Linear(128, classes, bias=True),
        # )
        if len(latent_dims) == 0 or latent_dims is None:
            self.fc_target = nn.Sequential(
                DropoutOrIdentity(dropout),
                nn.Linear(featureLength, classes, bias=True))

        else:
            indims = [featureLength] + latent_dims[:-1]
            outdims = latent_dims
            last_indim = outdims[-1] if len(outdims) > 0 else featureLength
            last_outdim = classes
            layers = []
            for indim, outdim in zip(indims, outdims):
                layers.append(nn.Linear(indim, outdim, bias=True))
                layers.append(DropoutOrIdentity(dropout))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(outdim))
            layers.append(nn.Linear(last_indim, last_outdim, bias=True))
            self.fc_target = nn.Sequential(*layers)

    def forward(self, x, race, lengths=None,return_features=False):
        features = self.featureExtractor(x.squeeze(0), lengths)
        preds = self.fc_target(features)
        if return_features:
            return preds, features
        else:
            return preds

class ClfNetwSensitive(ClfNet):
    def __init__(self, featureLength=768, classes=2, dropout=None):
        super(ClfNet, self).__init__()
        self.sens_projector = nn.Linear(1, featureLength, bias=True)

    def forward(self, x, race, lengths=None,return_features=False):
        # x_w_sensitive = torch.cat([x,race.unsqueeze(1)],1)
        sens_encoding = self.sens_projector(race)
        x_w_sensitive = x.squeeze(0) + sens_encoding.unsqueeze(1)
        
        features = self.featureExtractor(x_w_sensitive, lengths)
        features_w_sensitive = features + sens_encoding
        preds = self.fc_target(features)
        if return_features:
            return preds, features
        else:
            return preds



class MLP(nn.Module):
    def __init__(self, featureLength=768, 
                 latent_dims=[256, 128],
                 classes=2,  ft=False,dropout=None):
        super(MLP, self).__init__()

        # self.featureLength = featureLength
        print('featureLength')
        print(featureLength)

        if len(latent_dims) == 0 or latent_dims is None:
            self.fc_target = nn.Sequential(
                DropoutOrIdentity(dropout),
                nn.Linear(featureLength, classes, bias=True))

        else:
            indims = [featureLength] + latent_dims[:-1]
            outdims = latent_dims
            last_indim = outdims[-1] if len(outdims) > 0 else featureLength
            last_outdim = classes
            layers = []
            for indim, outdim in zip(indims, outdims):
                layers.append(nn.Linear(indim, outdim, bias=True))
                layers.append(DropoutOrIdentity(dropout))
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(outdim))
            layers.append(nn.Linear(last_indim, last_outdim, bias=True))
            self.fc_target = nn.Sequential(*layers)

    def forward(self, features, race, lengths=None,**kwargs):
        # features = self.featureExtractor(x.squeeze(0), lengths)
        preds = self.fc_target(features)
        if kwargs.get("return_features", False):
            return preds, features
        return preds
    
    
class MLPwSensitive(MLP):
    def __init__(self, featureLength=768, classes=2, ft=False,dropout=None):
        super(MLPwSensitive, self).__init__(featureLength=featureLength, classes=classes, ft=ft,dropout=dropout)
        self.sens_projector = nn.Linear(1, featureLength, bias=True)

    def forward(self, features, race, lengths=None, **kwargs):
        # features = self.featureExtractor(x.squeeze(0), lengths)
        # features_w_sensitive = torch.cat((features,race.unsqueeze(1)),1)
        sens_encoding = self.sens_projector(race.unsqueeze(1).float())
        features_w_sensitive = features + sens_encoding
        preds = self.fc_target(features_w_sensitive)
        if kwargs.get("return_features", False):
            return preds, features
        return preds