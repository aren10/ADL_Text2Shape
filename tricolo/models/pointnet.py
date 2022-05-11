from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(1024)
        self.bn4 = nn.InstanceNorm1d(512)
        self.bn5 = nn.InstanceNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(128)
        self.bn3 = nn.InstanceNorm1d(1024)
        self.bn4 = nn.InstanceNorm1d(512)
        self.bn5 = nn.InstanceNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeatMini(nn.Module):
    def __init__(self, input_k = 3, global_feat = True, feature_transform = False):
        super(PointNetfeatMini, self).__init__()
        #self.stn = STNkd(k=input_k)
        self.conv1 = torch.nn.Conv1d(input_k, 32, 1)
        self.conv2 = torch.nn.Conv1d(32, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        #self.bn1 = nn.BatchNorm1d(64)
        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        self.bn3 = nn.Identity()
        #self.bn2 = nn.BatchNorm1d(128)
        #self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[1]
        #trans = self.stn(x)
        x = x.transpose(2, 1)
        #x = torch.bmm(x, trans)
        #x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 128)
        if self.global_feat:
            #return x, trans, trans_feat
            return x
        else:
            x = x.view(-1, 128, 1).repeat(1, 1, n_pts)
            #return torch.cat([x, pointfeat], 1), trans, trans_feat
            return torch.cat([x, pointfeat], 1)

class PointNetfeat(nn.Module):
    def __init__(self, input_k = 3, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        #self.stn = STNkd(k=input_k)
        self.conv1 = torch.nn.Conv1d(input_k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.bn1 = nn.BatchNorm1d(64)
        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        self.bn3 = nn.Identity()
        #self.bn2 = nn.BatchNorm1d(128)
        #self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[1]
        #trans = self.stn(x)
        x = x.transpose(2, 1)
        #x = torch.bmm(x, trans)
        #x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            #return x, trans, trans_feat
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            #return torch.cat([x, pointfeat], 1), trans, trans_feat
            return torch.cat([x, pointfeat], 1)

class PointNetfeat2(nn.Module):
    def __init__(self, input_k = 19, global_feat = True, feature_transform = False):
        super(PointNetfeat2, self).__init__()
        #self.stn = STNkd(k=input_k)
        self.conv1 = torch.nn.Conv1d(input_k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(64)
        self.bn3 = nn.InstanceNorm1d(64)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[1]
        #trans = self.stn(x)
        x = x.transpose(2, 1)
        #x = torch.bmm(x, trans)
        #x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 64)
        if self.global_feat:
            #return x, trans, trans_feat
            return x
        else:
            x = x.view(-1, 64, 1).repeat(1, 1, n_pts)
            #return torch.cat([x, pointfeat], 1), trans, trans_feat
            return torch.cat([x, pointfeat], 1)

class PointNetClsMini(nn.Module):
    def __init__(self, input_k=19, k=2, feature_transform=False):
        super(PointNetClsMini, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeatMini(input_k = input_k, global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, k)
        self.dropout = nn.Dropout(p=0.3)
        #self.bn1 = nn.BatchNorm1d(512)
        #self.bn2 = nn.BatchNorm1d(256)
        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        #x, trans, trans_feat = self.feat(x)
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        #x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
        #return F.log_softmax(x, dim=1), trans, trans_feat

    def get_feature(self, x):
        x = self.feat(x)
        return x

class PointNetCls(nn.Module):
    def __init__(self, input_k=19, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(input_k = input_k, global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        #self.bn1 = nn.BatchNorm1d(512)
        #self.bn2 = nn.BatchNorm1d(256)
        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        #x, trans, trans_feat = self.feat(x)
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        #x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
        #return F.log_softmax(x, dim=1), trans, trans_feat

    def get_feature(self, x):
        x = self.feat(x)
        return x


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 16, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        #self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.feat = PointNetfeat2(input_k = 32, global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(128, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 32, 1)
        self.conv4 = torch.nn.Conv1d(32, self.k, 1)
        self.bn1 = nn.InstanceNorm1d(128)
        self.bn2 = nn.InstanceNorm1d(64)
        self.bn3 = nn.InstanceNorm1d(32)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[1]
        #x, trans, trans_feat = self.feat(x)
        x = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        #return x, trans, trans_feat
        return x

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
