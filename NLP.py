import pandas as pd
import torch
from torchtext.legacy import data
from tqdm import tqdm

tokenize = lambda x: x.split()
# sequential数据是不是按顺序来
# tokenize传入刚才切分好的数据
# lower小写
# fix_length补齐最大长度
# use_vocab是否使用Vocab对象。如果为False，则此字段中的数据应已为数字。默认值：True。
# 在这里其实就是获取分词后的文本和标签
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = data.Field(sequential=False, use_vocab=True)

# 构建数据集-自定义类
# 可以使用继承Dataset类的MyDatase
# 训练集、验证集、测试集的路径
train_path = './data/train.txt'
valid_path = './data/val.txt'
test_path = './data/test_data.txt'


# 定义Dataset
class MyDataset(data.Dataset):
    name = "Grand Dataset"

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, test=False):
        # 将文本中id、评论、标签放入field中
        fields = [("text", text_field),
                  ("label", label_field)]
        examples = []
        # 使用pandas读取数据，在这里传入的是一个path
        csv_data = pd.read_csv(train_path, delimiter=";", header=None, names=['text', 'label'])
        print('read data from {}'.format(path))
        # 对数据进行判断，如果是test数据，就不包含label内容，如果不是，就要读取toxic
        if test:
            for text in tqdm(csv_data['text']):
                examples.append(data.Example.fromlist([text, None], fields))
        else:
            for text, label in tqdm(zip(csv_data['text'], csv_data['label'])):
                examples.append(data.Example.fromlist([text, label], fields))
        # 之前是一些预处理操作，调用super调用父类构造方法，产生标准Dataset
        super(MyDataset, self).__init__(examples, fields)


# 构建数据集-构建数据集
train = MyDataset(train_path, text_field=TEXT, label_field=LABEL, test=False)
valid = MyDataset(valid_path, text_field=TEXT, label_field=LABEL, test=False)
test = MyDataset(test_path, text_field=TEXT, label_field=None, test=True)

# 构建词表
TEXT.build_vocab(train)
LABEL.build_vocab(train)

TEXT.build_vocab(valid)
LABEL.build_vocab(valid)
# 构建数据集迭代器

from torchtext.legacy.data import Iterator

# 好吧，源代码中是上面那么写的，在pycharm里面可以运行，但是在jupyter中报错，这里拆开来写了，效果一样
batch_size = 8
train_iter = data.BucketIterator(dataset=train, batch_size=batch_size, shuffle=True, sort_within_batch=False,
                                 repeat=False)
valid_iter = data.BucketIterator(dataset=valid, batch_size=batch_size, shuffle=True, sort_within_batch=False,
                                 repeat=False)

test_iter = Iterator(test, batch_size=batch_size, device=-1, sort=False, sort_within_batch=False, repeat=False)

# 接下来就是构造一个LSTM模型，然后训练一下
import torch.nn as nn
import torch.optim as optim


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.word_embedding = nn.Embedding(len(TEXT.vocab), 300)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 7)

    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        lstm_out = self.lstm(embeds)[0]
        final = lstm_out.sum(dim=0)
        y = self.fc1(final)
        y = self.fc2(y)
        return y


model = LSTM()
model.train()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
cri = torch.nn.CrossEntropyLoss()

epoch = 5
for ep in range(0, epoch, 1):
    total_loss = 0
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()
        predicted = model(batch.text)
        loss = cri(predicted, batch.label)
        loss.backward()
        optimizer.step()
        total_loss = loss
        if i % 200 == 0:
            print('{}th batch, loss is {}'.format(i, loss.item()))

    print("ep : {}, loss : {}".format(ep + 1, total_loss))
    with torch.no_grad():
        true_count = 0
        total_count = 0
        for i, valid_data in enumerate(valid_iter):
            text, label = valid_data.text, valid_data.label
            label1 = model(valid_data.text).argmax(-1)
            true_count += (label1 == label).sum()
            total_count += batch_size
        print("{}th epoch's accuracy is {}".format(ep + 1, true_count / total_count))

    # TODO predict test & print to .txt
    if ep == epoch - 1:
        # predict test
        pass
