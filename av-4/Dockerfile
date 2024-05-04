# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

# Copy the current directory contents into the container at /app
ADD author-identity.py /code/author-identity.py
ADD requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /code/requirements.txt

# Command to run the Python script
CMD ["python3", "author-identity.py"]
