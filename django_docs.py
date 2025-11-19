import argparse
import hashlib
import logging
import os
import pathlib
import pprint
import sys
import time

from chromadb.config import Settings
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredRSTLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import (
    CHROMA_DB_DIRECTORY,
    DJANGO_DOCS_DIR,
    DJANGO_COLLECTION_NAME,
    MODEL_EMBEDDINGS,
    MODEL_CHAT,
    VERSION,
)


logger = logging.getLogger(__name__)


def _get_vector_store():
    embeddings = OllamaEmbeddings(
        model=MODEL_EMBEDDINGS,
        base_url="http://localhost:11434",
    )

    # Initialise the vector store, but disable telemetry (sends to posthog).
    vector_store = Chroma(
        collection_name=f"{DJANGO_COLLECTION_NAME}_{VERSION}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIRECTORY,
        client_settings=Settings(anonymized_telemetry=False),
    )
    return vector_store


@dynamic_prompt
def _query_django_docs(request: ModelRequest) -> str:
    """Query django docs."""
    last_query = request.state["messages"][-1].text

    retrieved_docs = _similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # This could probably use some improvement (too verbose?).
    system_message = (
        "Act as a senior Python developer with extensive experience in software "
        "engineering. Provide expert guidance on best practices, design patterns, "
        "and performance optimization in Python applications. Offer insights on "
        "code structure, testing methodologies, and integration of APIs and "
        "libraries. If a specific question is presented, break down the problem, "
        "include code examples, offer critical thinking steps, and conclude "
        "with actionable recommendations. Use the following context as authoritative "
        "in your response:"
        f"\n\n{docs_content}"
    )

    return system_message

def _similarity_search(query: str, with_score: bool = False):
    vector_store = _get_vector_store()
    if with_score:
        return vector_store.similarity_search_with_score(query)
    else:
        return vector_store.similarity_search(query)


def search():
    query = input(f"Search for something related to Django ({VERSION}): ")

    # Include score.
    results = _similarity_search(query, True)

    separator = "\n====================\n"
    print(separator.join(doc.page_content for doc, _ in results))
    print("\n\nFound %s docs" % len(results))
    print("\n".join("%s %s" % (score, doc.metadata["source"]) for doc, score in results))


def ask():
    chat = ChatOllama(
        model=MODEL_CHAT,
        base_url="http://localhost:11434",
        temperature=0,
    )
    agent = create_agent(
        chat,
        tools=[],
        middleware=[_query_django_docs],
    )
    query = input(f"Ask a question related to Django ({VERSION}): ")

    start_time = time.time()

    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    total_seconds = time.time() - start_time
    logger.debug(f"Answered in {total_seconds} seconds\n\n")
    [x.pretty_print() for x in response["messages"]]


def load():
    vector_store = _get_vector_store()

    doc_files = []
    prefix_len = len(DJANGO_DOCS_DIR) + 1
    for path, subdirs, files in os.walk(DJANGO_DOCS_DIR):
        for name in files:
            filepath = os.path.join(path, name)[prefix_len:]
            if not filepath.startswith("releases") and name.endswith(".txt") and not filepath.startswith("_theme"):
                doc_files.append(filepath)

    doc_files = [x for x in doc_files if x not in ['index.txt', 'contents.txt', 'requirements.txt']]

    for index, doc_file in enumerate(doc_files):

        start_time = time.time()
        # Using mode of 'elements' seems to split way too much, e.g.
        #   67|You can also define logger namespacing explicitly:
        #   68|logger = logging.getLogger("project.payment")
        #   69|and set up logger mappings accordingly.
        # So let's try 'single' (the default). We're splitting into chunks later anyway.
        loader = UnstructuredRSTLoader(file_path=f"{DJANGO_DOCS_DIR}/{doc_file}", mode="single")
        documents = loader.load()
        logger.debug("%s documents loaded from %s (%s of %s)" % (len(documents), doc_file, index + 1, len(doc_files)))
        # Needed because chroma can't deal with lists (although it's possible this
        # is only present if we choose 'mode="elements") e.g. 'languages': ['eng']
        documents = filter_complex_metadata(documents)
        logger.debug("Documents filtered for complex metadata")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(documents)
        logger.debug("%s tokens prepared" % len(all_splits))
        filtered_splits = []
        # Let's add an id to each one based on source, start_index and page_content
        # so we avoid duplicating data if we rerun this.
        for split in all_splits:
            m = hashlib.sha256()
            m.update(split.metadata['source'].encode("utf-8"))
            m.update(str(split.metadata['start_index']).encode("utf-8"))
            m.update(split.page_content.encode("utf-8"))
            split.id = m.hexdigest()
            results = vector_store.get(ids=[split.id])
            if not results["documents"]:
                filtered_splits.append(split)
        logger.debug("%s tokens after filtering" % len(filtered_splits))

        if filtered_splits:
            ids = vector_store.add_documents(documents=filtered_splits)
            time_spent = time.time() - start_time
            time_per_token = time_spent / len(filtered_splits)
            logger.debug(f"Added {doc_file} to DB in {time_spent} seconds ({time_per_token} seconds per token)")


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser(
        prog="Django Docs",
        description="Query (or load) django docs",
    )
    parser.add_argument('action', choices=["ask", "load", "search"])
    args = parser.parse_args()
    if args.action == "load":
        load()
    elif args.action == "ask":
        ask()
    elif args.action == "search":
        search()
