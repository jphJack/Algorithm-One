import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine * self.s

        sine = torch.sqrt((1.0 - cosine * cosine).clamp(0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim=256, margin=0.5, scale=30.0, dropout=0.5):
        super(Classifier, self).__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(in_channels, embed_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.arc_margin = ArcMarginProduct(embed_dim, num_classes, s=scale, m=margin)

    def extract_embedding(self, x):
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = self.embedding_bn(x)
        x = self.dropout(x)
        return x

    def forward(self, x, labels=None, return_embedding=False):
        embedding = self.extract_embedding(x)
        logits = self.arc_margin(embedding, labels)
        if return_embedding:
            return logits, embedding
        return logits


if __name__ == '__main__':
    model = Classifier(in_channels=256, num_classes=290)

    x = torch.randn(2, 256, 7, 6)
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
