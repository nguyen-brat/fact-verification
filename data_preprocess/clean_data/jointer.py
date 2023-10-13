import json

class Jointer:
    def __init__(self, data_paths):
        self.data = []
        self.data_paths = data_paths
        for data_path in data_paths:
            with open(data_path, 'r') as f:
                self.data.append(json.load(f))
    
    def __call__(self, output_path):
        pass