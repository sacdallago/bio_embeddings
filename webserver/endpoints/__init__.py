from flask_restx import Api

api = Api(
    title="bio_embeddings API",
    description="Run bio_embedding pipeline jobs, or extract annotations via LMs synchronously."
)
