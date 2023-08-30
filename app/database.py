import redis, os

host = os.getenv("REDIS_HOST", "localhost")
port = int(os.getenv("REDIS_PORT", 6379))
db = int(os.getenv("REDIS_DB", 0))

pool = redis.ConnectionPool(host=host,
                            port=port,
                            db=db)

db = redis.Redis(connection_pool=pool)

def get_db():
    db = redis.Redis(connection_pool=pool)
    return db