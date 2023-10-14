from model.reranking.cross_encoder.trainer import rerank_run
from model.claim_verification.joint_cross_encoder.trainer import join_fact_verify_run
from inference.pipepine import Pipeline
import argparse

def parse_args():
    """
    Parse arguments from command line.
    """
    parser = argparse.ArgumentParser(description="Arguments for rerank Trainning")
    parser.add_argument("--train_type", default='rerank', type=str, help='specify the process want to run take only 3 value rerank, fact_verify, inferece')
    args = parser.parse_args()
    return args, parser

if __name__ == "__main__":
    args, parser = parse_args()
    if args.train_type == 'rerank':
        rerank_run(parser)
    elif args.train_type == 'fact_verify':
        join_fact_verify_run(parser)
    elif args.train_type == 'inferece':
        pipe = Pipeline()
        pipe.output_file(parser)
    else:
        print('not support this type')