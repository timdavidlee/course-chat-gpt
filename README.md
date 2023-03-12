# course-chat-gpt

Notes + materials from regarding `chat GPT`. Check the `mkdocs` hosted website here: [https://timdavidlee.github.io/course-chat-gpt/](https://timdavidlee.github.io/course-chat-gpt/)


## Setting up the python environment locally

Any version of python above 3.8 is fine

```sh
pyenv virtualenv 3.10.8 course-chat-gpt-3.10.8
echo 'course-chat-gpt-3.10.8' > .python-version
pyenv activate
```

To ensure the correct environment has been activated

```sh
which python
# >>> /Users/timlee/.pyenv/shims/python
pyenv which python
# >>> /Users/timlee/.pyenv/versions/course-chat-gpt-3.10.8/bin/python
```

Installing all the necessary python libraries:

```sh
pip install -r requirements.txt
```