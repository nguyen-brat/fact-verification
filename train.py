from model.reranking.trainer import rerank_run
from model.claim_verification.gear.trainer import fact_verify_run
from model.claim_verification.joint_cross_encoder.trainer import join_fact_verify_run

if __name__ == "__main__":
    #rerank_run()
    #fact_verify_run()
    join_fact_verify_run()
    pass