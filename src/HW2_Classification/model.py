import numpy as np
import torch
import random
import os

from torch import detach
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
import gc
from torch.utils.data.dataloader import DataLoader

#让实验可以浮现，种子相同就行
def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#加载文件中的特征张量
def load_feat(path):
    feat = torch.load(path)
    return feat

#对张量x进行位移操作
def shift(x,n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left,right),dim = 0)

#进行特征拼接
def concat_feat(x,concat_n):
    assert concat_n % 2 == 1
    if concat_n < 2:
        return x
    seq_len,feature_dim = x.size(0),x.xize(1)
    x = x.repeat(1,concat_n)
    x = x.view(seq_len,concat_n,feature_dim).permute(1,0,2)
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1,0,2).view(seq_len,concat_n * feature_dim)

#数据预处理
def preprocess_data(split,feat_dir,phone_path,concat_nframes,train_ratio = 0.8):
    class_num = 41 #不用改，提前算出来的

    if split =='train' or split == 'val':
        mode = 'train'

    elif split == 'test':
        mode = 'test'

    else:
        raise ValueError('Tnvalid \'split\'argument for dataset: PhoneDataset!')

    label_dict = {}

    if mode == 'train':
        for line in open(os.path.join(phone_path,f'{mode}_label.txt')).readlines():
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]

        usage_list = open(os.path.join(phone_path,'train_split.txt')).readlines()
        random.shuffle(usage_list)
        train_len = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:train_len] if split == 'train' else usage_list[train_len:]

    elif mode == 'test':
        usage_list = open(os.path.join(phone_path,'test_split.txt')).readlines()

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ',number of utterances for ' + split + ':' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode == 'train':
        y = torch.empty(max_len,dtype=torch.long)
    idx = 0
    for i,fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir,mode,f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode == 'train':
            lable = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode == 'trian':
            y[idx: idx + cur_len] = label_dict

        idx += cur_len

    X = X[:idx, :]
    if mode == 'train':
        y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode == 'train':
        print(y.shape)
        return X,y
    else :
        return X

class LibriDataset(Dataset):
    def __init__(self,X,y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.lable = None

    #魔术方法，是字典功能，返回键对应的值
    def __getitem__(self,idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len (self.data)

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock,self).__init__()

        #这里要实现对正则化和dropout的实现（dropout是方式过拟合的策略)
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html (batch normalization)
        #https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html (dropout)
        self.block = nn.Sequential(
            nn.Linear(input_dim,output_dim),
            nn.ReLU(),
        )
    def forward(self,x):
        x = self.block(x)
        return x

class Classifier(nn.MOdule):
    def __init(self, input_dim, output_dim = 41, hidden_layers = 1, hidden_dim = 256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)

        )

    def forward(self,x):
        x = self.fc(x)
        return x


#调参部分（调concat_nframes）
#concat_nframes必须是奇数
concat_nframes = 3
train_ratio = 0.75

seed = 42
batch_size = 512
num_epoch = 10
learning_rate = 1e-4
#模型保存的地址
model_path = './model.ckpt'

##调参部分（调hidden_layers或者hidden_dim）
input_dim = 39 * concat_nframes
hidden_layers = 2
hidden_dim = 64

same_seeds(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

train_X,train_y = preprocess_data(split = 'train', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes,train_ratio=train_ratio)
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes,train_ratio=train_ratio)

train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

del train_X, train_y, val_X, val_y
gc.collect()

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle = True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle = False)

model = Classifier(input_dim=input_dim, hidden_layers = hidden_layers, hidden_dim = hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for i,batch in enumerate(tqdm(train_loader)):
        featrues, labels = batch
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _,train_pred = torch.max(outputs,1)
        train_acc += (train_pred,detach() == labels.detach()).sum().item()
        train_loss += loss.item()

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)

            loss = criterion(outputs, labels)

            _, val_pred = torch.max(outputs,1)
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
            val_loss += loss.item()


    print(f'[{epoch+1 :03d}/{num_epoch:03d}] Train Acc : {train_acc/len(train_set):3.5f} Loss: {train_loss/len(train_loader):3.5f} | Val Acc: {val_acc/len(val_set):3.5f} loss : {val_loss/len(val_loader):3.5f}')

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(),model_path)
        print(f'saving model with acc {best_acc/len(val_set):.5f}')


del train_set, val_set
del train_loader, val_loader
gc.collect()

test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes)
test_set = LibriDataset(test_X,None)
test_loader = DataLoader(test_set,batch_size = batch_size,shuffle=False)

model = Classifier(input_dim = input_dim, hiddeen_layers = hidden_layers, hidden_dim = hidden_dim).to(device)
model.load_state_dict(torch.load(model_path))

pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _,test_pred = torch.max(outputs, 1)
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis= 0)

with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i ,y in enumerate(pred):
        f.write('{},{}\n'.format(i,y))