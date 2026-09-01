"""Central config. Override anything via environment variables."""
import os

# --- AWS ---
REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("RAG_S3_BUCKET", "")  # set to your free-tier bucket
S3_PREFIX = "longitudinal-chart-rag"
SM_EXECUTION_ROLE = os.environ.get("SM_EXECUTION_ROLE", "")  # SageMaker exec role ARN

# --- Models ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"          # 384-dim, CPU-friendly
EMBED_INSTANCE = "ml.m5.xlarge"                  # CPU, for the Processing Job
BEDROCK_MODEL = "anthropic.claude-haiku-4-5"     # generation

# --- Retrieval ---
TOP_K = 5
RECALL_KS = (1, 3, 5)
CHUNK_STRATEGY = "encounter"                     # "encounter" | "fixed512"

# --- Cohort / eval ---
N_PATIENTS_TARGET = 30
SYNTHEA_OVERGEN = 200
MEASURE = "CMS122"                               # HbA1c poor control (>9%)

# --- Local paths (gitignored) ---
DATA_DIR = os.environ.get("RAG_DATA_DIR", "data")
ARTIFACT_DIR = os.environ.get("RAG_ARTIFACT_DIR", "artifacts")
