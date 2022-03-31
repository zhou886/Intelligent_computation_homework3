from torch.utils.data import Dataset
import json
import torch

class MyDataset(Dataset):
    def __init__(self, train:bool = True) -> None:
        super().__init__()
        self.train = train
        if train:
            file = open("./data/train.json",'r', encoding="utf-8")
        else:
            file = open("./data/test.json", 'r', encoding="utf-8")
        tmp_list = file.readlines()
        self.size = len(tmp_list)
        self.data = []
        for iter in tmp_list:
            self.data.append(json.loads(iter))
        with open('./word_to_id.txt', 'r') as f:
            self.dirctionary = json.loads(f.read())
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, index:int) -> tuple:
        output1 = torch.zeros(120, dtype=int)
        output2 = torch.zeros(120, dtype=int)
        for i in range(len(self.data[index]['sentence1'])):
            output1[i] = self.dirctionary[self.data[index]['sentence1'][i]]
        for i in range(len(self.data[index]['sentence2'])):
            output2[i] = self.dirctionary[self.data[index]['sentence2'][i]]
        output3 = torch.tensor(int(self.data[index]['label']), dtype=float)
        return output1, output2, output3

    def showData(self) -> None:
        for i in range(self.size):
            o1, o2, o3 = self[i]
            print(o1, o2, o3)
    
    def showMaxLength(self) -> None:
        maxLen = 0
        for i in range(self.size):
            s1 = self[i][0]
            s2 = self[i][1]
            maxLen = max(len(s1), len(s2), maxLen)
        print("Max sequence length is", maxLen)

if __name__=="__main__":
    dataset = MyDataset()
    dataset.showData()