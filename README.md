## Install

### Ollama

Install Ollama, and the deepseek-r1:1.5b and nomic-embed-text:latest models locally.

### Django docs

You'll need a local copy of the Django documentation. Optionally you can switch to the branch you want for a specific
version of the docs (e.g. `origin/stable/4.2.x`).
`git clone https://github.com/django/django`
You can also have other documentation types, e.g. Laravel. See the example settings file for more details.

### Dependencies

Set up a python virtualenv and install the dependencies from `requirements.txt`.

### Settings

Copy `settings_example.py` to `example.py` and set all the variables as appropriate.

## Run

To load the docs into Chroma DB, run `python3 docs.py load django`. This may well take quite a while, depending on the
speed of your setup.

To run a search directly against the Chroma DB, run `python3 docs.py search django`.

To ask a query of deepseek, taking into account context from Chroma DB, run `python3 docs.py ask django`.

To compare the output of this setup with a raw query, ask the same question (make sure you mention it's for the version
of Django or Laravel you're building docs for to ollama directly.

## Motivation

The goal of this project is to be able to use AI in a local-only set up with minimal resources on a very specific data
set (in this case, the Django or Laravel documentation).
