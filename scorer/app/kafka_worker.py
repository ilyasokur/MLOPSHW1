import os, json, asyncio, pandas as pd
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from app.scripts.score import load_model_bundle
import json as _json, os as _os
def _load_feature_names(model_dir: str):
    p=_os.path.join(model_dir,"feature_names.json")
    if _os.path.exists(p):
        with open(p,"r",encoding="utf-8") as f: return _json.load(f)
    return None

from app.scripts.preprocess_adapter import preprocess_event

BOOT = os.getenv("KAFKA_BOOTSTRAP","kafka:29092")
IN_TOPIC = os.getenv("INPUT_TOPIC","transactions")
OUT_TOPIC = os.getenv("OUTPUT_TOPIC","scores")
THRESH = float(os.getenv("THRESHOLD","0.7"))

MODEL_DIR = "/workspace/app/model"
model, _ = load_model_bundle(MODEL_DIR)
feature_names = _load_feature_names(MODEL_DIR)

def _predict_scores(X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:,1]
    return model.predict(X)

async def run():
    consumer = AIOKafkaConsumer(
        IN_TOPIC, bootstrap_servers=BOOT, enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")))
    producer = AIOKafkaProducer(
        bootstrap_servers=BOOT,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"))
    await consumer.start(); await producer.start()
    try:
        async for msg in consumer:
            rec = msg.value
            tx_id = rec.get("index") or rec.get("transaction_id") or rec.get("id")
            X = preprocess_event(rec)
            if feature_names:
                cols = [c for c in feature_names if c in X.columns]
                X = X[cols]
            score = float(_predict_scores(X)[0])
            flag = int(score >= THRESH)
            out = {"transaction_id": str(tx_id), "score": score, "fraud_flag": flag}
            await producer.send_and_wait(OUT_TOPIC, out)
    finally:
        await consumer.stop(); await producer.stop()

if __name__ == "__main__":
    asyncio.run(run())