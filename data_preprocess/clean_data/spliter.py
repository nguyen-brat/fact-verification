import json
import random
import os
import copy

class Spliter:
    def __init__(self, data_path):
        self.data_path = data_path
        with open(data_path, 'r') as f:
            self.data = list(json.load(f).values())
            random.shuffle(self.data)
        self.data_length = len(self.data)

    def split_balance(self, output_path, num_sample_per_label:int=100):
        '''
        split into valid and train set number of sample per label equal to pram num_sample_per_label
        '''
        data = copy.deepcopy(self.data)
        valid_result = {}
        train_result = {}
        refuted_count = num_sample_per_label
        supported_count = num_sample_per_label
        nei_count = num_sample_per_label
        for i, sample in enumerate(self.data):
            if (refuted_count > 0) and (sample['verdict']=="REFUTED"):
                valid_result[str(i)] = sample
                refuted_count -= 1
                data.pop(i)
            elif (nei_count > 0) and (sample['verdict']=="NEI"):
                valid_result[str(i)] = sample
                nei_count -= 1
                data.pop(i)
            elif (supported_count > 0) and (sample['verdict']=="SUPPORTED"):
                valid_result[str(i)] = sample
                supported_count -= 1
                data.pop(i)
            if (refuted_count + supported_count + nei_count) == 0:
                break
        for i in range(self.data_length - 3*num_sample_per_label):
            train_result[str(i)] = data[i]
        
        with open(os.path.join(output_path, 'train.json'), 'w') as f:
            json.dump(train_result, f, ensure_ascii=False, indent=4)
        with open(os.path.join(output_path, 'valid.json'), 'w') as f:
            json.dump(valid_result, f, ensure_ascii=False, indent=4)

    def split_random(self, output_path, valid_size:float=0.1):
        '''
        split into valid and train set number of sample is
        '''
        train_result = {}
        valid_result = {}
        train = self.data[int(self.data_length*valid_size):]
        valid = self.data[:int(self.data_length*valid_size)]
        for i in range(len(train)):
            train_result[str(i)] = train[i]
        for i in range(len(valid)):
            valid_result[str(i)] = valid[i]
        with open(os.path.join(output_path, 'train.json'), 'w') as f:
            json.dump(train_result, f, ensure_ascii=False, indent=4)
        with open(os.path.join(output_path, 'valid.json'), 'w') as f:
            json.dump(valid_result, f, ensure_ascii=False, indent=4)
        return None
    
if __name__ == '__main__':
    spliter = Spliter('hidden/ise-dsc01-train.json')
    #spliter.split_random('hidden')
    spliter.split_balance('hidden')