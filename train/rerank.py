import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.extend([parent_dir])
from model.reranking.cross_encoder.trainer import rerank_run


if __name__ == "__main__":
    rerank_run()
    pass