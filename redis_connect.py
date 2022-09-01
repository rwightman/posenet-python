import redis

def redis_init()
    global redis_pool
    redis_pool = redis.ConnectionPool(host = , port = decode_responses = True)

def read_data(key)
    redis_conn = redis.Redis(connection_pool = redis_pool)
    return redis_conn.get(key)

def set_data(key, value)
    redis_conn = redis.Redis(connection_pool = redis_pool)
    redis_conn.set(key, value)
    