#!/usr/bin/env bash
# deploy.sh — Train → Build → Push → Redeploy → Verify
# Usage: ./scripts/deploy.sh [--skip-train] [--skip-verify]
#
# Requires: aws-cli v2, docker, uv
# AWS credentials must be active in the environment before running.

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
AWS_REGION="us-east-2"
AWS_ACCOUNT="321305251356"
ECR_REPO="predmaint-api"
IMAGE_TAG="latest"
TASK_FAMILY="predmaint-api"
DOCKERFILE="Dockerfile.api"
MODEL_PATH="models/model.pkl"

ECR_URI="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

SKIP_TRAIN=false
SKIP_VERIFY=false

# ─── Argument parsing ─────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --skip-train)   SKIP_TRAIN=true ;;
    --skip-verify)  SKIP_VERIFY=true ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--skip-train] [--skip-verify]"
      exit 1
      ;;
  esac
done

# ─── Helpers ──────────────────────────────────────────────────────────────────
log()  { echo "[deploy] $*"; }
fail() { echo "[deploy] ERROR: $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" &>/dev/null || fail "'$1' not found. Please install it first."
}

# ─── Pre-flight checks ────────────────────────────────────────────────────────
log "Running pre-flight checks..."
require_cmd aws
require_cmd docker
require_cmd uv

aws sts get-caller-identity --query "Account" --output text &>/dev/null \
  || fail "No valid AWS credentials found. Run 'aws configure' or export AWS_* env vars."

log "AWS identity: $(aws sts get-caller-identity --query 'Arn' --output text)"

# ─── Step 1: Train the model ──────────────────────────────────────────────────
if [ "$SKIP_TRAIN" = false ]; then
  log "Step 1/5 — Training the model..."
  uv run python src/training/train.py
  [ -f "$MODEL_PATH" ] || fail "Training finished but $MODEL_PATH not found."
  log "Model saved at $MODEL_PATH ($(du -sh "$MODEL_PATH" | cut -f1))"
else
  log "Step 1/5 — Skipping training (--skip-train flag set)."
  [ -f "$MODEL_PATH" ] || fail "$MODEL_PATH not found. Remove --skip-train to train first."
fi

# ─── Step 2: ECR login ────────────────────────────────────────────────────────
log "Step 2/5 — Authenticating with ECR..."
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# ─── Step 3: Build Docker image ───────────────────────────────────────────────
log "Step 3/5 — Building Docker image..."
docker build \
  --file "$DOCKERFILE" \
  --tag "${ECR_REPO}:${IMAGE_TAG}" \
  --tag "$ECR_URI" \
  .

# ─── Step 4: Push to ECR ─────────────────────────────────────────────────────
log "Step 4/5 — Pushing image to ECR..."
docker push "$ECR_URI"
log "Pushed: $ECR_URI"

# ─── Step 5: Force ECS redeploy ───────────────────────────────────────────────
log "Step 5/5 — Discovering ECS cluster and service..."

# Find the cluster that runs tasks from this task family
CLUSTER=$(aws ecs list-clusters --region "$AWS_REGION" --query "clusterArns[]" --output text \
  | tr '\t' '\n' \
  | while read -r arn; do
      # Check if any service in this cluster uses our task family
      SVC=$(aws ecs list-services --cluster "$arn" --region "$AWS_REGION" \
        --query "serviceArns[]" --output text 2>/dev/null || true)
      if [ -n "$SVC" ]; then
        for svc_arn in $SVC; do
          TASK_DEF=$(aws ecs describe-services \
            --cluster "$arn" --services "$svc_arn" \
            --region "$AWS_REGION" \
            --query "services[0].taskDefinition" --output text 2>/dev/null || true)
          if [[ "$TASK_DEF" == *"${TASK_FAMILY}"* ]]; then
            echo "$arn"
            break 2
          fi
        done
      fi
    done)

[ -n "$CLUSTER" ] || fail "Could not find an ECS cluster running task family '${TASK_FAMILY}'."
log "Found cluster: $CLUSTER"

SERVICE=$(aws ecs list-services --cluster "$CLUSTER" --region "$AWS_REGION" \
  --query "serviceArns[]" --output text \
  | tr '\t' '\n' \
  | while read -r svc_arn; do
      TASK_DEF=$(aws ecs describe-services \
        --cluster "$CLUSTER" --services "$svc_arn" \
        --region "$AWS_REGION" \
        --query "services[0].taskDefinition" --output text 2>/dev/null || true)
      if [[ "$TASK_DEF" == *"${TASK_FAMILY}"* ]]; then
        echo "$svc_arn"
        break
      fi
    done)

[ -n "$SERVICE" ] || fail "Could not find an ECS service using task family '${TASK_FAMILY}'."
log "Found service: $SERVICE"

log "Forcing new deployment..."
aws ecs update-service \
  --cluster "$CLUSTER" \
  --service "$SERVICE" \
  --force-new-deployment \
  --region "$AWS_REGION" \
  --query "service.deployments[0].status" \
  --output text

log "Waiting for service to become stable (this may take ~2 minutes)..."
aws ecs wait services-stable \
  --cluster "$CLUSTER" \
  --services "$SERVICE" \
  --region "$AWS_REGION"
log "Service is stable."

# ─── Step 6: Verify endpoint ──────────────────────────────────────────────────
if [ "$SKIP_VERIFY" = false ]; then
  log "Verify — Resolving public IP of the new task..."

  # Get the task ARN for the newly deployed task
  TASK_ARN=$(aws ecs list-tasks \
    --cluster "$CLUSTER" \
    --region "$AWS_REGION" \
    --query "taskArns[0]" \
    --output text 2>/dev/null || true)

  PUBLIC_IP=""
  if [ -n "$TASK_ARN" ] && [ "$TASK_ARN" != "None" ]; then
    ENI_ID=$(aws ecs describe-tasks \
      --cluster "$CLUSTER" \
      --tasks "$TASK_ARN" \
      --region "$AWS_REGION" \
      --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value | [0]" \
      --output text 2>/dev/null || true)

    if [ -n "$ENI_ID" ] && [ "$ENI_ID" != "None" ]; then
      PUBLIC_IP=$(aws ec2 describe-network-interfaces \
        --network-interface-ids "$ENI_ID" \
        --region "$AWS_REGION" \
        --query "NetworkInterfaces[0].Association.PublicIp" \
        --output text 2>/dev/null || true)
    fi
  fi

  if [ -z "$PUBLIC_IP" ] || [ "$PUBLIC_IP" = "None" ]; then
    log "WARNING: Could not determine task public IP automatically."
    log "Run manually:"
    log "  curl http://<TASK_PUBLIC_IP>:8000/health"
    log "  curl -X POST http://<TASK_PUBLIC_IP>:8000/predict -H 'Content-Type: application/json' \\"
    log "    -d '{\"air_temperature\":298.1,\"process_temperature\":308.6,\"rotational_speed\":1551,\"torque\":42.8,\"tool_wear\":0,\"type_h\":0,\"type_l\":1,\"type_m\":0}'"
    exit 0
  fi

  BASE_URL="http://${PUBLIC_IP}:8000"
  log "Task public endpoint: $BASE_URL"

  log "Checking GET /health..."
  for i in 1 2 3 4 5; do
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/health" || true)
    if [ "$HTTP_STATUS" = "200" ]; then
      log "/health -> HTTP $HTTP_STATUS OK"
      break
    fi
    log "Attempt $i: /health returned HTTP $HTTP_STATUS. Retrying in 10s..."
    sleep 10
  done

  if [ "$HTTP_STATUS" != "200" ]; then
    fail "/health returned $HTTP_STATUS after 5 attempts. Check CloudWatch logs: /ecs/predmaint-api"
  fi

  log "Testing POST /predict with a sample payload..."
  PREDICT_RESPONSE=$(curl -s -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" \
    -d '{
      "air_temperature": 298.1,
      "process_temperature": 308.6,
      "rotational_speed": 1551,
      "torque": 42.8,
      "tool_wear": 0,
      "type_h": 0,
      "type_l": 1,
      "type_m": 0
    }')

  echo ""
  log "/predict response:"
  echo "$PREDICT_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$PREDICT_RESPONSE"
  echo ""

  FAILURE_PREDICTED=$(echo "$PREDICT_RESPONSE" | python3 -c \
    "import sys, json; d=json.load(sys.stdin); print(d.get('failure_predicted','MISSING'))" 2>/dev/null || true)

  if [ "$FAILURE_PREDICTED" = "MISSING" ]; then
    fail "Unexpected /predict response. Check CloudWatch logs: /ecs/predmaint-api"
  fi

  log "Deploy complete. failure_predicted=$FAILURE_PREDICTED"
  log "Endpoint: $BASE_URL"
else
  log "Verification skipped (--skip-verify flag set)."
  log "To get the current task IP:"
  log "  aws ecs list-tasks --cluster predmaint-cluster --region us-east-2"
fi
