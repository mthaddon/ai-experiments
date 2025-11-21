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
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    UnstructuredRSTLoader,
)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import (
    CHROMA_DB_DIRECTORY,
    MODELS,
    SOURCES,
)


logger = logging.getLogger(__name__)


def _get_vector_store(source):
    embeddings = OllamaEmbeddings(
        model=MODELS["embeddings"]["name"],
        base_url=MODELS["embeddings"]["url"],
    )

    collection_name = SOURCES[source]["collection_name"]
    version = SOURCES[source]["version"]
    # Initialise the vector store, but disable telemetry (sends to posthog).
    vector_store = Chroma(
        collection_name=f"{collection_name}_{version}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIRECTORY,
        client_settings=Settings(anonymized_telemetry=False),
    )
    return vector_store


@dynamic_prompt
def _query_docs(request: ModelRequest) -> str:
    """Query docs."""
    last_query = request.state["messages"][-1].text
    source = request.runtime.context.get("source", "django")

    retrieved_docs = _similarity_search(source, last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    if source == "django":
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
    elif source == "laravel":
        system_message = (
            "Act as a senior PHP developer with extensive experience in software "
            "engineering. Provide expert guidance on best practices, design patterns, "
            "and performance optimization in PHP applications. Offer insights on "
            "code structure, testing methodologies, and integration of APIs and "
            "libraries. If a specific question is presented, break down the problem, "
            "include code examples, offer critical thinking steps, and conclude "
            "with actionable recommendations. Use the following context as authoritative "
            "in your response:"
            f"\n\n{docs_content}"
        )
    else:
        raise Exception("Unknown source type in _query_docs")

    return system_message

def _similarity_search(source: str, query: str, with_score: bool = False):
    vector_store = _get_vector_store(source)
    if with_score:
        return vector_store.similarity_search_with_score(query)
    else:
        return vector_store.similarity_search(query)


def search(source):
    prog = SOURCES[source]["name"]
    version = SOURCES[source]["version"]
    query = input(f"Search for something related to {prog} ({version}): ")

    # Include score.
    results = _similarity_search(source, query, True)

    separator = "\n====================\n"
    print(separator.join(doc.page_content for doc, _ in results))
    print("\n\nFound %s docs" % len(results))
    print("\n".join("%s %s" % (score, doc.metadata["source"]) for doc, score in results))


def ask(source):
    prog = SOURCES[source]["name"]
    version = SOURCES[source]["version"]
    chat = ChatOllama(
        model=MODELS["chat"]["name"],
        base_url=MODELS["chat"]["url"],
        temperature=0,
    )
    agent = create_agent(
        chat,
        tools=[],
        middleware=[_query_docs],
    )
    query = input(f"Ask a question related to {prog} ({version}): ")

    start_time = time.time()

    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        context={"source": source}
    )

    total_seconds = time.time() - start_time
    logger.debug(f"Answered in {total_seconds} seconds\n\n")
    [x.pretty_print() for x in response["messages"]]


def load(source):
    vector_store = _get_vector_store(source)
    docs_dir = SOURCES[source]["docs_dir"]
    doc_type = SOURCES[source]["doc_type"]

    doc_files = []
    prefix_len = len(docs_dir) + 1
    for path, subdirs, files in os.walk(docs_dir):
        for name in files:
            filepath = os.path.join(path, name)[prefix_len:]
            if source == "django":
                if not filepath.startswith("releases") and name.endswith(".txt") and not filepath.startswith("_theme"):
                    doc_files.append(filepath)
            elif source == "laravel":
                if filepath.endswith(".md"):
                    doc_files.append(filepath)

    if source == "django":
        doc_files = [x for x in doc_files if x not in ['index.txt', 'contents.txt', 'requirements.txt']]

    for index, doc_file in enumerate(doc_files):

        start_time = time.time()
        # Using mode of 'elements' seems to split way too much, e.g.
        #   67|You can also define logger namespacing explicitly:
        #   68|logger = logging.getLogger("project.payment")
        #   69|and set up logger mappings accordingly.
        # So let's try 'single' (the default). We're splitting into chunks later anyway.
        if doc_type == "rst":
            loader = UnstructuredRSTLoader(file_path=f"{docs_dir}/{doc_file}", mode="single")
        elif doc_type == "md":
            loader = UnstructuredMarkdownLoader(file_path=f"{docs_dir}/{doc_file}", mode="single")
        else:
            raise Exception("Unknown doc_type")
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
    parser.add_argument('source', choices=["django", "laravel"])
    args = parser.parse_args()
    if args.action == "load":
        load(args.source)
    elif args.action == "ask":
        ask(args.source)
    elif args.action == "search":
        search(args.source)
