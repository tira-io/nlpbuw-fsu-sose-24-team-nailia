# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM nelliemirz/buw-nlp-bert-base-uncased:0.0.1

# Copy the current directory contents into the container at /app
ADD author-identity.py /code/author-identity.py
# ADD ./bert-base-uncased/. /code/bert-base-uncased/

# Command to run the Python script
ENTRYPOINT ["python3", "/code/author-identity.py"]
