# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM nelliemirz/buw-nlp-bert-base-uncased:latest

# Copy the current directory contents into the container at /app
ADD author-identity.py /code/author-identity.py

# Command to run the Python script
ENTRYPOINT ["python3", "/code/author-identity.py"]
