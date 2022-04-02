from torch.utils.data import Dataset
import json
import torch


class MyDataset(Dataset):
    def __init__(self, train: bool = True) -> None:
        super().__init__()
        self.train = train
        if train:
            file = open("./data/train.json", 'r', encoding="utf-8")
        else:
            file = open("./data/dev.json", 'r', encoding="utf-8")
        tmp_list = file.readlines()
        self.size = len(tmp_list)
        self.data = []
        for iter in tmp_list:
            self.data.append(json.loads(iter))
        with open('./word_to_id.json', 'r') as f:
            self.dirctionary = json.loads(f.read())

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple:
        """
        return sentence1, sentence2, label, length of sentence1, length of sentence2
        """
        s1 = self.data[index]['sentence1']
        s2 = self.data[index]['sentence2']
        output1 = torch.zeros(90, dtype=int)
        output2 = torch.zeros(90, dtype=int)
        for i in range(len(s1)):
            if s1[i] in self.dirctionary:
                output1[i] = self.dirctionary[s1[i]]
        for i in range(len(s2)):
            if s2[i] in self.dirctionary:
                output2[i] = self.dirctionary[s2[i]]
        output3 = torch.tensor(int(self.data[index]['label']), dtype=float)
        return output1, output2, output3, len(s1), len(s2)

    def showData(self) -> None:
        for i in range(self.size):
            print(self.data[i]['sentence1'], self.data[i]
                  ['sentence2'], self.data[i]['label'])

    def countSentenceLength(self) -> None:
        length = {}
        for i in range(self.size):
            s1 = self.data[i]['sentence1']
            s2 = self.data[i]['sentence2']
            if len(s1) > 90 or len(s2) > 90:
                print("too long", i)
            if len(s1) not in length:
                length[len(s1)] = 1
            else:
                length[len(s1)] += 1
            if len(s2) not in length:
                length[len(s2)] = 1
            else:
                length[len(s2)] += 1
        for key in length:
            print(key, length[key])

    def makeDictionary(self) -> None:
        dictionary = {}
        for i in range(self.size):
            s1 = self.data[i]['sentence1']
            s2 = self.data[i]['sentence2']
            for ch in s1+s2:
                if ch not in dictionary:
                    dictionary[ch] = len(dictionary)+1
        with open('./word_to_id.json', 'w') as f:
            json.dump(dictionary, f)


if __name__ == "__main__":
    dataset = MyDataset()
    dataset.makeDictionary()
