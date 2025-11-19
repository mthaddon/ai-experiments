# The directory where Chroma DB will reside. Must exist beforehand.
CHROMA_DB_DIRECTORY="/home/ubuntu/repos/ai-experiments/chroma_db"
# The directory where the django docs can be found. Note that this is the docs
# directory with the repo gh:django/django
DJANGO_DOCS_DIR="/home/ubuntu/repos/ai-experiments/sources/django/docs"
# Than prefix for the Chroma DB collection.
DJANGO_COLLECTION_NAME="django_docs"
# The model to use for embeddings (i.e. creating numeric representations of
# text that will then be used to query similarity to input text).
MODEL_EMBEDDINGS="nomic-embed-text:latest"
# The model to use for inference.
MODEL_CHAT="deepseek-r1:1.5b"
# The version of the django documentation to use. This should correspond to
# the git branch you've downloaded to DJANGO_DOCS_DIR.
VERSION="4.2"
