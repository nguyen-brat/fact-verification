from haystack.nodes import DensePassageRetriever
from haystack.utils import fetch_archive_from_http
from haystack.document_stores import InMemoryDocumentStore
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
if __name__ == "__main__":
    train_filename = "train/biencoder-nq-train.json"
    dev_filename = "dev/biencoder-nq-dev.json"
    doc_dir='data'
    
    query_model = "bert-base-uncased"
    passage_model = "bert-base-uncased"

    save_dir = "../saved_models/dpr"

    ## Initialize DPR model

    retriever = DensePassageRetriever(
        document_store=InMemoryDocumentStore(),
        query_embedding_model=query_model,
        passage_embedding_model=passage_model,
        max_seq_len_query=64,
        max_seq_len_passage=256,
    )

    retriever.train(
        data_dir=doc_dir,
        train_filename=train_filename,
        dev_filename=dev_filename,
        test_filename=dev_filename,
        n_epochs=1,
        batch_size=16,
        grad_acc_steps=8,
        save_dir=save_dir,
        evaluate_every=3000,
        embed_title=True,
        num_positives=1,
        num_hard_negatives=1,
    )
