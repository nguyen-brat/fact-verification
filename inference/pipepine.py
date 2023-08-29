from haystack.pipelines import Pipeline
from haystack.nodes import BM25Retriever, EmbeddingRetriever, SentenceTransformersRanker
from .custom_node import fact_verification, reranking
from haystack.nodes import JoinDocuments

def fact_checking_pipline(
        document_store,
        embed_model,
):
    # Initialize Sparse Retriever
    bm25_retriever = BM25Retriever(document_store=document_store)

    # Initialize embedding Retriever
    embedding_retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model=embed_model
    )
    pipeline = Pipeline()
    pipeline.add_node(component=bm25_retriever, name="Retriever", inputs=["Query"])
    pipeline.add_node(component=embedding_retriever, name="Reader", inputs=["Query"])
    pipeline.add_node(
        component=JoinDocuments(join_mode="concatenate"), name="JoinResults", inputs=["BM25Retriever", "EmbeddingRetriever"]
    )
    pipeline.add_node(component=reranking, name='reranking', input=["JoinResults"])
    pipeline.add_node(component=fact_verification, name="verification", inputs=["reranking"])

    return pipeline


if __name__ == "__main__":
    pass