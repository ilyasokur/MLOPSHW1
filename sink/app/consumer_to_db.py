import os, json, asyncio, psycopg2
from aiokafka import AIOKafkaConsumer

BOOT = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
IN_TOPIC = os.getenv("INPUT_TOPIC", "scores")
PG_DSN = os.getenv("PG_DSN", "postgresql://fraud:fraud@postgres:5432/fraud")

conn = None
def get_conn():
    global conn
    if conn is None or conn.closed:
        conn = psycopg2.connect(
            PG_DSN,
            keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5,
        )
        conn.autocommit = True
    return conn

def insert_row(txid, score, flag):
    c = get_conn()
    with c.cursor() as cur:
        print(f"[sink] inserting: tx={txid} score={score} flag={flag}")
        cur.execute(
            "INSERT INTO scores (transaction_id, score, fraud_flag) VALUES (%s,%s,%s)",
            (str(txid), float(score), int(flag)),
        )

async def run():
    consumer = AIOKafkaConsumer(
        IN_TOPIC,
        bootstrap_servers=BOOT,
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    await consumer.start()
    try:
        async for msg in consumer:
            d = msg.value
            print(f"[sink] received: {d}")
            insert_row(d.get("transaction_id"), d.get("score"), d.get("fraud_flag"))
    finally:
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(run())