from flask_restplus import Api

api = Api(
    title="Embedding extractor API",
    description="Extract Elmo embeddings from sequences"
)