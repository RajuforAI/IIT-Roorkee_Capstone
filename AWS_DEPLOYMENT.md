# AWS Deployment Guide

> **Push TeleGenie AI to production on AWS**
> Two deployment paths: **AWS App Runner** (recommended, single-container) and **Amazon ECS on Fargate** (enterprise-grade, multi-service). Pick one.

---

## Table of Contents

1. [Architecture Decision](#1-architecture-decision)
2. [Account & Permissions Setup](#2-account--permissions-setup)
3. [Path A — AWS App Runner (recommended)](#3-path-a--aws-app-runner-recommended)
4. [Path B — Amazon ECS on Fargate (fallback)](#4-path-b--amazon-ecs-on-fargate-fallback)
5. [Secret Management — SSM Parameter Store + Secrets Manager](#5-secret-management)
6. [Observability — CloudWatch + LangSmith](#6-observability)
7. [Cost Expectations](#7-cost-expectations)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Architecture Decision

| Criteria | **App Runner** ⭐ recommended | **ECS Fargate** |
|---|---|---|
| Time to first deploy | ~15 min | ~45 min |
| Monthly cost (low traffic) | **~$5–25** | ~$50–100 |
| Scales to zero | ❌ (min instance always on) | ✅ |
| Custom VPC / private subnets | ❌ | ✅ |
| Multi-container (app + Chroma) | ⚠️ single container | ✅ |
| TLS / HTTPS | ✅ automatic | ✅ via ALB |
| CI/CD from GitHub | ✅ built-in | ⚠️ via CodePipeline |
| Best for | **Capstone demo, single-container Streamlit** | Enterprise, private networking |

**Recommendation:** Path A (App Runner) unless you have a specific reason to need private subnets, scale-to-zero, or to run Chroma as a separate service.

---

## 2. Account & Permissions Setup

### 2.1 IAM user for deployment

Create an IAM user (or role) with these managed policies:

- `AmazonAppRunnerFullAccess` (Path A)
- `AmazonECS_FullAccess` (Path B)
- `AmazonS3FullAccess`
- `SecretsManagerReadWrite`
- `AmazonSSMFullAccess`
- `CloudWatchLogsFullAccess`
- `AWSCloudFormationFullAccess`

Tag the user with a clear `Purpose=telegenie-deploy` so you can audit later.

### 2.2 Configure the AWS CLI

```powershell
aws configure
# AWS Access Key ID: <your access key>
# AWS Secret Access Key: <your secret key>
# Default region: us-east-1   (or your preferred region)
# Default output format: json
```

Verify:

```powershell
aws sts get-caller-identity
```

---

## 3. Path A — AWS App Runner (recommended)

App Runner pulls your GitHub repo, builds the Dockerfile, runs the container, provisions HTTPS, and auto-scales. **Zero infrastructure to manage.**

### 3.1 Push the repo to GitHub

```powershell
cd IIT-Roorkee_Capstone
git remote add origin git@github.com:RajuforAI/IIT-Roorkee_Capstone.git
git push -u origin main
```

> The repo's `apprunner.yaml` (in the repo root) is auto-discovered by App Runner — it pins the runtime, port, build, and start commands. You don't need to re-enter them in the console.

### 3.2 Create the App Runner service

**Via Console (easiest):**

1. AWS Console → **App Runner** → **Create service**
2. **Source:** GitHub → Connect repository → `RajuforAI/IIT-Roorkee_Capstone` → branch `main` → ✅ "Use a configuration file" (it will read `apprunner.yaml`)
3. **Service name:** `telegenie-ai`
4. **Virtual environment:** Python 3, build command `pip install -r requirements.txt`, start command `streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0`
5. **Service settings:**
   - vCPU: 1 vCPU
   - Memory: 3 GB (LangGraph + Chroma need headroom)
   - Port: `8501`
6. **Auto-scaling:**
   - Min: 1 instance
   - Max: 4 instances
   - Concurrency: 50 requests per instance
7. **Health check:** Path `/` (Streamlit responds 200)
8. **Security:** IAM role = "Create new service role"
9. **Tags:** `Project=telegenie-ai`, `Environment=production`, `Owner=rajubera`
10. Click **Create & deploy** — first build takes ~5–8 minutes.

**Via AWS CLI:**

```powershell
aws apprunner create-service `
  --service-name telegenie-ai `
  --source-configuration file://apprunner-source.json `
  --instance-configuration file://apprunner-instance.json `
  --health-check-configuration file://apprunner-healthcheck.json `
  --tags Key=Project,Value=telegenie-ai Key=Environment,Value=production
```

See `cloudformation/apprunner-*.json` for the JSON files referenced above.

### 3.3 Wire secrets (App Runner → SSM + Secrets Manager)

In the App Runner console, under **Service → Configuration → Environment variables**, add:

| Key | Value source |
|---|---|
| `TELECOM_RAG_OPENAI_API_KEY` | Secrets Manager: `telegenie/openai-api-key` |
| `TELECOM_RAG_GEMINI_API_KEY` | Secrets Manager: `telegenie/gemini-api-key` |
| `TELECOM_RAG_LANGCHAIN_API_KEY` | Secrets Manager: `telegenie/langchain-api-key` |
| `TELECOM_RAG_SECRET_KEY` | SSM Parameter: `/telegenie/secret-key` (SecureString) |
| `TELECOM_RAG_AWS_ACCESS_KEY_ID` | SSM Parameter: `/telegenie/aws-access-key-id` (SecureString) |
| `TELECOM_RAG_AWS_SECRET_ACCESS_KEY` | SSM Parameter: `/telegenie/aws-secret-access-key` (SecureString) |
| `TELECOM_RAG_AWS_S3_BUCKET` | Plain value: `telegenie-ai-prod-docs` |

App Runner natively reads these from SSM Parameter Store and Secrets Manager at deploy time. Reference syntax:

- **SSM:** `arn:aws:ssm:us-east-1:123456789012:parameter/telegenie/secret-key`
- **Secrets Manager:** `arn:aws:secretsmanager:us-east-1:123456789012:secret:telegenie/openai-api-key`

In the App Runner console, when you add an env var, click **"Add from secret"** and paste the ARN.

See [§5](#5-secret-management) for the full secret-bootstrap script.

### 3.4 Create the S3 bucket for PDF uploads

```powershell
aws s3api create-bucket `
  --bucket telegenie-ai-prod-docs `
  --region us-east-1

# Block all public access
aws s3api put-public-access-block `
  --bucket telegenie-ai-prod-docs `
  --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

# Enable versioning (for audit trail)
aws s3api put-bucket-versioning `
  --bucket telegenie-ai-prod-docs `
  --versioning-configuration Status=Enabled
```

### 3.5 Verify the deploy

App Runner will print a public URL like `https://xyz123.us-east-1.awsapprunner.com` when the service is `Running`.

```powershell
# Health check
curl https://xyz123.us-east-1.awsapprunner.com/_stcore/health
# → "ok"

# Open in browser
start https://xyz123.us-east-1.awsapprunner.com
```

You should see the Streamlit login screen.

### 3.6 Subsequent deploys

Just `git push` to `main`. App Runner watches the branch and redeploys automatically (default: auto-deploy on push; toggle in console).

---

## 4. Path B — Amazon ECS on Fargate (fallback)

Use this if you need private subnets, scale-to-zero, or to run ChromaDB as a separate service. Provisioned via CloudFormation — see `cloudformation/template.yaml`.

### 4.1 What gets created

| Resource | Purpose |
|---|---|
| **VPC + 2 public subnets + 2 private subnets** | Network isolation |
| **ALB (Application Load Balancer)** | HTTPS termination, path routing |
| **ECS Cluster** (`telegenie-cluster`) | Fargate cluster |
| **ECS Service** (`telegenie-service`) | Runs the Streamlit task, desired count 2 |
| **Task Definition** | 1 vCPU / 3 GB, port 8501, reads from ECR |
| **Security Groups** | Least-privilege ingress/egress |
| **CloudWatch Log Group** | `/ecs/telegenie-ai` — 30-day retention |
| **Secrets** | SSM + Secrets Manager entries (see §5) |
| **S3 Bucket** | `telegenie-ai-prod-docs` |

### 4.2 Build and push the image to ECR

```powershell
$AWS_REGION = "us-east-1"
$ECR_REPO = "telegenie-ai"
$ECR_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com/telegenie-ai"

# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin "$ECR_URI"

# Create the ECR repo (one-time)
aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION

# Build and tag
docker build -t $ECR_REPO:latest .
docker tag $ECR_REPO:latest "${ECR_URI}:latest"

# Push
docker push "${ECR_URI}:latest"
```

### 4.3 Deploy the CloudFormation stack

```powershell
aws cloudformation deploy `
  --template-file cloudformation/template.yaml `
  --stack-name telegenie-ai-prod `
  --capabilities CAPABILITY_IAM `
  --parameter-overrides `
      EnvironmentName=production `
      ImageUri="${ECR_URI}:latest" `
      ContainerPort=8501 `
      DesiredCount=2 `
      DomainName=telegenie.example.com `
      CertificateArn=arn:aws:acm:us-east-1:123456789012:certificate/xxxxx
```

### 4.4 Verify

The stack outputs an `ALBDNSName` (something like `telegenie-alb-1234567890.us-east-1.elb.amazonaws.com`).

```powershell
# Health check
curl http://<ALBDNSName>/_stcore/health

# Tail logs
aws logs tail /ecs/telegenie-ai --follow
```

---

## 5. Secret Management

All secrets live in **AWS Secrets Manager** (API keys) or **AWS Systems Manager Parameter Store** (non-secret config), and are referenced by ARN from App Runner / ECS task definitions. Nothing is hardcoded.

### 5.1 Bootstrap script

Run this **once** before deploying:

```powershell
$AWS_REGION = "us-east-1"

# ---- Secrets Manager (API keys) ----
aws secretsmanager create-secret `
  --name telegenie/openai-api-key `
  --secret-string "REPLACE_WITH_OPENAI_KEY" `
  --region $AWS_REGION

aws secretsmanager create-secret `
  --name telegenie/gemini-api-key `
  --secret-string "REPLACE_WITH_GEMINI_KEY" `
  --region $AWS_REGION

aws secretsmanager create-secret `
  --name telegenie/langchain-api-key `
  --secret-string "REPLACE_WITH_LANGCHAIN_KEY" `
  --region $AWS_REGION

# ---- SSM Parameter Store (non-secret config) ----
aws ssm put-parameter `
  --name "/telegenie/secret-key" `
  --value "REPLACE_WITH_32_BYTE_HEX" `
  --type SecureString `
  --region $AWS_REGION

aws ssm put-parameter `
  --name "/telegenie/aws-access-key-id" `
  --value "REPLACE_WITH_AWS_ACCESS_KEY" `
  --type SecureString `
  --region $AWS_REGION

aws ssm put-parameter `
  --name "/telegenie/aws-secret-access-key" `
  --value "REPLACE_WITH_AWS_SECRET_KEY" `
  --type SecureString `
  --region $AWS_REGION
```

### 5.2 Rotation policy

Set up automatic rotation for the OpenAI / Gemini keys every 90 days:

```powershell
aws secretsmanager rotate-secret `
  --secret-id telegenie/openai-api-key `
  --rotation-lambda-arn arn:aws:lambda:us-east-1:123456789012:function:telegenie-rotate `
  --rotation-rules "AutomaticallyAfterDays=90"
```

(The rotation Lambda is out of scope for this README — but the entry point is `secretsmanager rotate-secret`.)

### 5.3 IAM for App Runner / ECS

The service's task role must have:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"], "Resource": "arn:aws:secretsmanager:*:*:secret:telegenie/*" },
    { "Effect": "Allow", "Action": ["ssm:GetParameters", "ssm:GetParameter"], "Resource": "arn:aws:ssm:*:*:parameter/telegenie/*" },
    { "Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"], "Resource": "arn:aws:s3:::telegenie-ai-prod-docs/*" },
    { "Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"], "Resource": "arn:aws:logs:*:*:log-group:/ecs/telegenie-ai:*" }
  ]
}
```

---

## 6. Observability

### CloudWatch Logs

- **App Runner:** logs auto-stream to `/aws/apprunner/telegenie-ai/<service-id>` — view in CloudWatch → Log groups.
- **ECS:** logs stream to `/ecs/telegenie-ai` (30-day retention by default).
- **Custom metrics:** the app emits `requests_total`, `latency_seconds`, `llm_tokens_total`, `llm_cost_usd` via embedded metric filters in the JSONL audit log.

### LangSmith tracing

Set these env vars (via Secrets Manager):

```
TELECOM_RAG_LANGCHAIN_TRACING_V2=true
TELECOM_RAG_LANGSMITH_PROJECT=TeleGenie AI
TELECOM_RAG_LANGSMITH_ENDPOINT=https://api.smith.langchain.com
TELECOM_RAG_LANGCHAIN_API_KEY=<from Secrets Manager>
```

Then open https://smith.langchain.com → project "TeleGenie AI" to see every graph run.

### Alarms (recommended)

```powershell
# CPU > 70% for 5 min → email
aws cloudwatch put-metric-alarm `
  --alarm-name telegenie-cpu-high `
  --metric-name CPUUtilization `
  --namespace AWS/AppRunner `
  --statistic Average `
  --period 300 `
  --threshold 70 `
  --comparison-operator GreaterThanThreshold `
  --evaluation-periods 1 `
  --alarm-actions arn:aws:sns:us-east-1:123456789012:ops-alerts

# 5xx responses > 10 in 5 min → email
aws cloudwatch put-metric-alarm `
  --alarm-name telegenie-5xx-spike `
  --metric-name 5xxResponse `
  --namespace AWS/AppRunner `
  --statistic Sum `
  --period 300 `
  --threshold 10 `
  --comparison-operator GreaterThanThreshold `
  --evaluation-periods 1 `
  --alarm-actions arn:aws:sns:us-east-1:123456789012:ops-alerts
```

---

## 7. Cost Expectations

Assumes a low-traffic capstone demo (~1k requests/day, average 1.5k tokens/response).

| Service | Monthly cost (us-east-1) |
|---|---|
| App Runner (1 instance × 1 vCPU × 3 GB, always on) | ~$25 |
| S3 (10 GB storage, 100k requests) | ~$1 |
| Secrets Manager (10 secrets) | ~$4 |
| CloudWatch Logs (5 GB ingestion + 30-day retention) | ~$3 |
| Data transfer out (50 GB) | ~$5 |
| **Total App Runner path** | **~$38/month** |
| + OpenAI API (1k requests × 1.5k tokens × $0.15/1M output) | **~$0.50** |
| + ChromaDB (in-process, free) | $0 |
| + LangSmith (free tier: 5k traces/month) | $0 |

**Capstone demo total: under $45/month.** Scale up for production traffic — most of the cost shifts to OpenAI API usage as you grow.

---

## 8. Troubleshooting

### Build fails: "pip install" timeout

Increase App Runner's build timeout (console → Service → Configuration → Edit → Build → Timeout to 30 min). The first build with PyTorch-shaped wheels can take 8+ minutes.

### Container starts but health check fails

Streamlit's default health endpoint is `/_stcore/health`. App Runner's health-check path must match. Check the Streamlit logs in CloudWatch:

```powershell
aws logs tail /aws/apprunner/telegenie-ai/<service-id> --follow
```

### "TELECOM_RAG_OPENAI_API_KEY is required" at startup

The Secrets Manager ARN is wrong, or the IAM role lacks `secretsmanager:GetSecretValue`. Re-check:

```powershell
aws secretsmanager get-secret-value --secret-id telegenie/openai-api-key
```

### Out of memory (OOM killed)

Increase the task memory from 3 GB to 4 GB (App Runner: Service → Configuration → Instance → Memory). ChromaDB + LangGraph + Streamlit session state together can push 3 GB.

### Slow first request after idle

Expected — LangChain re-initializes on first request. App Runner keeps the container warm by default; if you enable scale-to-zero, the first request after idle takes ~5–10 seconds.

---

## Appendix: Quick Reference

| Command | Purpose |
|---|---|
| `aws apprunner list-services` | List App Runner services |
| `aws apprunner describe-service --service-arn <arn>` | Service status + URL |
| `aws ecs list-services --cluster telegenie-cluster` | List ECS services |
| `aws ecs update-service --cluster telegenie-cluster --service telegenie-service --force-new-deployment` | Force ECS redeploy |
| `aws logs tail /aws/apprunner/telegenie-ai/<id> --follow` | Tail App Runner logs |
| `aws s3 sync chroma_db/ s3://telegenie-ai-prod-docs/backups/chroma_db/` | Manual Chroma backup |
| `docker compose down && docker compose up -d --build` | Local reset |

---

**Last updated:** June 2026 · TeleGenie AI v1.0.0