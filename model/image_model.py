import torch
import torchvision.models as models
from torch import nn
from text_model import TextPreprocessor, load_tweet_text_data, load_image_text_data


class InceptionV3FeatureExtractor(nn.Module):
    # 构造函数，声明模型的组成
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        self.inception_v3 = models.inception_v3(pretrained=True)
        print(self.inception_v3)

    def forward(self, x):
        for name, module in self.inception_v3.named_children():
            if name == 'AuxLogits':
                continue
            x = module(x)
            if name == 'avgpool':
                return x
        return x




if __name__ == '__main__':
    # model = InceptionV3FeatureExtractor()
    # input_tensor = torch.randn(1, 3, 512, 512) # (batch_size, channel, height, width)
    #
    # out = model(input_tensor)
    # print(out.shape)
    # model = LSTMNet()
    pass

