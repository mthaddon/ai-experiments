# The directory where Chroma DB will reside. Must exist beforehand.
CHROMA_DB_DIRECTORY="/home/ubuntu/repos/ai-experiments/chroma_db"
# The models to use for chat and embeddings (generating integers from text for
# later querying in the Chroma DB).
MODELS = {
    "chat": "deepseek-r1:1.5b",
    "embeddings": "nomic-embed-text:latest",
}
# The sources to use for ingestion and querying.
SOURCES = {
    "django": {
        "name": "Django",
        "collection_name": "django_docs",
        "doc_type": "rst",
        "docs_dir": "/home/ubuntu/repos/ai-experiments/sources/django/docs",
        "version": "4.2",
    },
    "laravel": {
        "name": "Laravel",
        "collection_name": "laravel_docs",
        "doc_type": "md",
        "docs_dir": "/home/ubuntu/repos/ai-experiments/sources/laravel/docs",
        "version": "11.x",
    }
}
