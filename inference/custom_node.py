from haystack import BaseComponent
from typing import Optional, List
from model.claim_verification.gear.model import fact_verification
from sentence_transformers.cross_encoder import CrossEncoder
from haystack.nodes import BM25Retriever, FARMReader

class reranking(BaseComponent):
    outgoing_edges = 1
    def __init__(self, path):
        self.model = CrossEncoder(path)

    def run(self, query: str, inputs:List[str]):
        output = self.model(
            query,
            inputs,
        )
        return output, "output_1"

    def run_batch(self, queries: List[str], inputs:List[List[str]]):
        # process the inputs
        output = {"my_output": ...}
        return output, "output_1"

class fact_verification(BaseComponent):
    outgoing_edges = 1
    def __init__(self, path):
        self.model = fact_verification.from_pretrained(path)
    def run(self, query: str, my_optional_param: Optional[int]):
        # process the inputs
        output = {"my_output": ...}
        return output, "output_1"

    def run_batch(self, queries: List[str], my_optional_param: Optional[int]):
        # process the inputs
        output = {"my_output": ...}
        return output, "output_1"

if __name__ == '__main__':
    pass