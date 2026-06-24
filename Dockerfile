FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x run.sh

# Default command if none is provided
CMD ["./run.sh", "./candidates.jsonl", "./submission.csv"]
