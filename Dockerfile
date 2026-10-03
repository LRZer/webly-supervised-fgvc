# Added during repository organization; not a recovered competition image.
FROM python:3.11-slim
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "scripts/validate_predictions.py", "results/predictions/pred_results_web400.csv", "--num-classes", "400", "--expected-rows", "5687"]
