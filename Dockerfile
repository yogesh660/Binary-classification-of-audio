# Dockerfile - this is a comment. Delete me if you want.
FROM python:3.7.6
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN apt-get update && \
      apt-get -y --no-install-recommends install sudo && \
      sudo apt-get -y --no-install-recommends install libsndfile1-dev && \
      pip install --no-cache-dir -r requirements.txt && \
      sudo rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["python"]
CMD ["app.py"]

