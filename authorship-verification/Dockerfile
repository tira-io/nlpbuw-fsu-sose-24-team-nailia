# docker build -t fschlatt/authorship-verification-trivial:0.0.1 .
FROM fschlatt/natural-language-processing-exercises:0.0.1

ADD authorship_verification.py /code/authorship_verification.py
ADD train_lr.py /code/train_lr.py

ENTRYPOINT [ "python3", "/code/authorship_verification.py" ]
