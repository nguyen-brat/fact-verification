from haystack.nodes import DensePassageRetriever
from haystack.utils import fetch_archive_from_http
from haystack.document_stores import InMemoryDocumentStore
from .model import BiEncoder
'''
    {
        "dataset": str,
        "question": str,
        "answers": list of str
        "positive_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
        "negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
        "hard_negative_ctxs": list of dictionaries of format {'title': str, 'text': str, 'score': int, 'title_score': int, 'passage_id': str}
    }
'''

class Trainer:
    def __init__(
            self,
            model,
    ):
        self.model = BiEncoder(model, model)


if __name__ == "__main__":
    pass
