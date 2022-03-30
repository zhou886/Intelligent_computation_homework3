from torch.utils.data import Dataset
import json

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
    
    def __getitem__(self, index:int) -> tuple:
        return self.data[index]['sentence1'], self.data[index]['sentence2'], self.data[index]['label']
    
    def getDictionary(self) -> dict:
        dictionary = {}
        for i in range(self.size):
            tmp = self[i]
            s1 = tmp[0]
            s2 = tmp[1]
            for ch in s1:
                if ch not in dictionary:
                    dictionary[ch] = len(dictionary)
            for ch in s2:
                if ch not in dictionary:
                    dictionary[ch] = len(dictionary)

        return dictionary

    def showData(self) -> None:
        for i in range(self.size):
            print(self[i])

if __name__=="__main__":
    dataset = MyDataset()
    word_to_id = dataset.getDictionary()
    with open("./word_to_id.txt", "w") as f:
        json.dump(word_to_id, f)