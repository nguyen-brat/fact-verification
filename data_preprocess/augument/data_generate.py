import os
from ..clean_data.file_processing import Jointer, Spliter, duplicate
from ..clean_data.preprocess import CleanData

def data_generate(original_data_path='hidden/origin/ise-dsc01-train.json', output_path='hidden/main'):
    spliter = Spliter(original_data_path)
    spliter.split_balance('hidden/postfile')
    
    duplicate(input_paths='hidden/postfile/train.json', output_path='hidden/postfile/refuted.json')

    jointer = Jointer(['hidden/postfile/train.json']+['hidden/postfile/refuted.json']*7)
    jointer('hidden/postfile/blance_trained_raw_data.json')

    val_cleaner = CleanData(data_path='hidden/postfile/valid.json')
    train_cleaner = CleanData(data_path='hidden/postfile/blance_trained_raw_data.json')
    val_cleaner(output_path=output_path+'val.json')
    train_cleaner(output_path=output_path+'train.json')

if __name__ == '__main__':
    pass