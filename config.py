"""Central config. Override anything via environment variables."""
import os

# --- AWS ---
REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("RAG_S3_BUCKET", "")  # set to your free-tier bucket
S3_PREFIX = "longitudinal-chart-rag"
SM_EXECUTION_ROLE = os.environ.get("SM_EXECUTION_ROLE", "")  # SageMaker exec role ARN

# --- Models ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"          # 384-dim, CPU-friendly
EMBED_INSTANCE = "ml.t3.medium"                  # CPU, for the Processing Job
BEDROCK_MODEL = "anthropic.claude-haiku-4-5"     # generation

# --- Retrieval ---
TOP_K = 5
RECALL_KS = (1, 3, 5)
CHUNK_STRATEGY = "section"                       # "section" (split >MAX on headers) | "fixed512"
MAX_CHUNK_TOKENS = 512                           # bge-small max sequence length
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge-small retrieval
INDEX_SOURCE = os.environ.get("RAG_INDEX_SOURCE", "local")  # "local" | "s3"

# --- Cohort / eval ---
N_PATIENTS_TARGET = 30
SYNTHEA_OVERGEN = 300
SYNTHEA_STATE = "Massachusetts"
SYNTHEA_SEED = 42
MEASURE = "CMS122"                               # HbA1c poor control (>9%)
DIABETES_SNOMED = "44054006"                     # Diabetes mellitus type 2
HBA1C_LOINC = "4548-4"                           # Hemoglobin A1c
HBA1C_CONTROL_THRESHOLD = 9.0                    # <= 9% == controlled
MIN_HBA1C_YEARS = 2                              # years in the denominator to keep a patient

# --- Local paths (gitignored) ---
DATA_DIR = os.environ.get("RAG_DATA_DIR", "data")
ARTIFACT_DIR = os.environ.get("RAG_ARTIFACT_DIR", "artifacts")
