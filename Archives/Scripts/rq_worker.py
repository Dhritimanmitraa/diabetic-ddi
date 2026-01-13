"""
RQ worker entry point.

Run with:
    cd backend
    python -m scripts.rq_worker
"""
import os
from rq import Worker, Queue, Connection
import redis

listen = ["default"]
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

conn = redis.from_url(redis_url)


def main():
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()


if __name__ == "__main__":
    main()

