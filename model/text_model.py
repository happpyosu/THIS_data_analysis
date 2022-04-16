import json
import os
from gensim.models import word2vec


def load_image_text_data():
    base_dir = '../MMHS150K/img_txt/'
    json_file_list = os.listdir(base_dir)
    x = []
    for file_name in json_file_list:
        with open(base_dir + file_name) as file:
            d = json.loads(file.readline())
            x.append(d['img_text'])

    return x


def load_tweet_text_data():
    with open('../MMHS150K/MMHS150K_GT.json') as file:
        d = json.loads(file.readline())

    x = []
    for tweet_id, val_dict in d.items():
        tweet_text = val_dict['tweet_text']
        tweet_text = str(tweet_text).split("https")[0]
        x.append(tweet_text)

    return x


def train_word2vec(x):
    model = word2vec.Word2Vec(x, sg=1)
    return model


def do_train():
    save_path = '../save/w2v.model'
    image_text = load_image_text_data()
    tweet_text = load_tweet_text_data()
    all_text = image_text + tweet_text

    print('all text length is: {}'.format(len(all_text)))

    # word2vec_model = train_word2vec(all_text)
    # word2vec_model.save(save_path)


if __name__ == '__main__':
    do_train()
