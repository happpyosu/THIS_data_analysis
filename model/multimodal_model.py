from image_model import InceptionV3FeatureExtractor
from text_model import LSTMNet, TextPreprocessor, load_tweet_text_data, load_image_text_data
import torch
from torch import nn


def get_embedding_matrix():
    tweet_text = load_tweet_text_data()
    image_text = load_image_text_data()
    return TextPreprocessor(tweet_text + image_text).make_embedding()


class FCMModel(nn.Module):
    def __init__(self):
        super(FCMModel, self).__init__()
        # inception v3 image model
        self.inception_v3 = InceptionV3FeatureExtractor()

        # lstm model
        self.lstm = LSTMNet(get_embedding_matrix())

        self.fc_0 = nn.Linear(2048+150+150, 1024)
        # todo

    def forward(self, img, image_text, tweet_text):
        # image feature with dim of 2048
        out_img = self.inception_v3(img)

        # image_text feature with dim of 300, where 150 for image_text and 150 for tweet_text
        text_input = torch.cat([image_text, tweet_text], dim=0)
        out_text = self.lstm()



