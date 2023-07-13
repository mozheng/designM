from torch.utils.data import Dataset
import torch
 
 
class trainDataset(Dataset):
    def __init__(self,layers_files, masks_files):
        # 读入全部数据
        self.data = torch.Tensor([[[1.0], [0]], [[2.0], [0]], [[3.0], [1]], [[4.0], [2]], [[5.0], [3]]])
        print('执行了init方法', '\n')
 
    def __getitem__(self, index):
        # 给定index 返回其数据
        # print("执行了getitem方法, 返回数据 ", self.data[index], '\n')
        return self.data[index]
 
    def __len__(self):
        print('执行了len方法, 返回', self.data.shape[0], '\n')
        return self.data.shape[0]
