import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.extend([parent_dir])
from model.claim_verification.joint_cross_encoder.trainer import join_fact_verify_run


if __name__ == "__main__":
    join_fact_verify_run()
    pass