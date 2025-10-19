set -euo pipefail

echo "[1/4] Loading input..."
python -m app.scripts.load_input --input_dir /mnt/input --work_dir /mnt/work --config app/config/config.yaml

echo "[2/4] Preprocessing..."
python -m app.scripts.preprocess --work_dir /mnt/work --config app/config/config.yaml

echo "[3/4] Scoring..."
python -m app.scripts.score --work_dir /mnt/work --model_dir app/model --config app/config/config.yaml

echo "[4/4] Exporting outputs..."
python -m app.scripts.export_outputs --work_dir /mnt/work --output_dir /mnt/output --config app/config/config.yaml

echo "Done. Check /mnt/output"