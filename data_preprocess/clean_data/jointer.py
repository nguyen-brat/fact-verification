import json
from typing import List

class Jointer:
    def __init__(self, data_paths:List):
        self.data = []
        self.data_paths = data_paths
        for data_path in data_paths:
            with open(data_path, 'r') as f:
                self.data.append(json.load(f))
    
    def __call__(self, output_path):
        '''
        join sample of all data_path and save in output_path
        '''
        result = {}
        count = 0
        for data in self.data:
            for sample in data.values():
                result[str(count)] = sample
                count += 1
        with open(output_path, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    jointer = Jointer(['data/raw_data/ise-dsc01-warmup.json', 'data/raw_data/Back_translate_augmentation_data.json', 'data/raw_data/Paraphrase_augmentation_data.json'])
    jointer('hidden/raw_data.json')