import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_softmax_pooling(x):
    return (x ** 2).sum(1) / x.sum(1)


class Crnn(nn.Module):
    def __init__(self, num_freq, num_class):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     num_freq: int, mel frequency bins
        #     num_class: int, the number of output classes
        ##############################
        super(Crnn,self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_freq)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1,16,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2)),

            nn.Conv2d(16,32,kernel_size =3,stride = 1,padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2,2)),

            nn.Conv2d(32,64,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1,2)),

            nn.Conv2d(64,128,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size = (1,2)),

            nn.Conv2d(128,128,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size = (1,2))
        )
        self.gru = nn.GRU(128,128,bidirectional = True,batch_first=True)
        # self.lstm = nn.LSTM(128,128,bidirectional = True,batch_first=True)
        # self.rnn = nn.RNN(128,128,bidirectional = True,batch_first =True)
        # self.gru = nn.GRU(512,128,bidirectional = True,batch_first=True)

        # self.fc1 = nn.Linear(256,128)
        # self.fc2 = nn.Linear(128,64)
        # self.fc3 = nn.Linear(64,num_class)
        
        self.fc = nn.Linear(256,num_class)
        # self.dropout = nn.dropout(p = 0.2)


    def detection(self, x):
        ##############################
        # YOUR IMPLEMENTATION
        # Args:
        #     x: [batch_size ,time_steps, num_freq]
        # Return:
        #     frame_prob: [batch_size ,time_steps, num_class]
        ##############################
        # x = x.to('cuda')
        x = x.permute(0, 2, 1)
        x = self.batch_norm1(x)
        x = x.permute(0, 2, 1)
        # torch.Size([64, 501, 64])
        x = x.unsqueeze(1)
        # torch.Size([64,1 ,501, 64])
        
        x = self.conv_layers(x)
        x = x.mean(dim=3)  
        # x = F.interpolate(x,scale_factor=4,mode = 'linear')
        x = x.permute(0, 2, 1)
        
        x, _ = self.gru(x)
        # x, _ = self.lstm(x)
        # x,_ = self.rnn(x)

        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        x = self.fc(x)

        x = torch.sigmoid(x)
        x = F.interpolate(x.permute(0,2,1),scale_factor=4,mode = 'linear')
        x = x.permute(0,2,1)
        return x

    def forward(self, x):
        frame_prob = self.detection(x)  # (batch_size, time_steps, class_num)
        clip_prob = linear_softmax_pooling(frame_prob)  # (batch_size, class_num)
        '''(samples_num, feature_maps)'''
        return {
            'clip_prob': clip_prob, 
            'frame_prob': frame_prob
        }
