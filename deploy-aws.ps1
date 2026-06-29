# ============================================================================
# TeleGenie AI  - AWS Bootstrap Script
# ============================================================================
# Idempotent: safe to re-run. Sections with side effects use --overwrite or
# catch 409 conflicts. Tested against AWS CLI 2.x.
#
# Usage:
#   .\deploy-aws.ps1 -AwsAccountId 123456789012 -AwsRegion us-east-1
#
# Prerequisites:
#   - AWS CLI 2.x installed and configured (`aws configure`)
#   - IAM permissions: secretsmanager:*, ssm:*, s3:*, ecr:*, iam:*, apprunner:*, logs:*
#   - GitHub repo connected (App Runner will pull from it)
# ============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$AwsAccountId,
    [Parameter(Mandatory=$true)][string]$AwsRegion = 'us-east-1',
    [Parameter(Mandatory=$true)][string]$EnvironmentName = 'production',
    [Parameter(Mandatory=$false)][string]$GitHubRepo = 'RajuforAI/IIT-Roorkee_Capstone',
    [Parameter(Mandatory=$false)][string]$GitHubBranch = 'main',
    [Parameter(Mandatory=$false)][switch]$SkipApprunner,  # for testing only
    [Parameter(Mandatory=$false)][switch]$CreateCustomApprunnerRole,
    [Parameter(Mandatory=$false)][switch]$WhatIfMode
)

# ============================================================================
# Constants
# ============================================================================
$ErrorActionPreference = 'Continue'
$ServiceName           = 'telegenie-ai'
$DocsBucketName        = "telegenie-ai-${EnvironmentName}-docs-${AwsAccountId}"
$EcrRepoName           = 'telegenie-ai'
$LogGroupName          = "/aws/apprunner/telegenie-ai/${EnvironmentName}"
$InstanceRoleName      = 'telegenie-apprunner-instance-role'
$SecretNames           = @(
    'telegenie/openai-api-key',
    'telegenie/gemini-api-key',
    'telegenie/langchain-api-key'
)
$SsmParamNames         = @(
    '/telegenie/secret-key',
    '/telegenie/aws-access-key-id',
    '/telegenie/aws-secret-access-key'
)
$CreatedSecrets        = @()
$ServiceArn            = $null
$ServiceUrl            = $null

# ============================================================================
# Helpers
# ============================================================================
# Unicode glyphs emitted via [char] so the file stays pure ASCII and the
# PowerShell parser (which reads via the current ANSI codepage) doesn't choke.
$CheckGlyph = [char]0x2713   # check mark
$CrossGlyph = [char]0x2717   # cross
$WarnGlyph  = [char]0x26A0   # warning sign
$ArrowGlyph = [char]0x2192   # rightwards arrow

function Write-Step {
    param([string]$Message, [string]$Status = 'info')
    switch ($Status) {
        'success' { Write-Host ("$CheckGlyph {0}" -f $Message) -ForegroundColor Green }
        'failure' { Write-Host ("$CrossGlyph {0}" -f $Message) -ForegroundColor Red }
        'warn'    { Write-Host ("$WarnGlyph  {0}" -f $Message) -ForegroundColor Yellow }
        default   { Write-Host ("$ArrowGlyph {0}" -f $Message) -ForegroundColor Cyan }
    }
}

function Invoke-Aws {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)][string]$Description,
        [Parameter(Mandatory=$true)][scriptblock]$ScriptBlock
    )
    try {
        $result = & $ScriptBlock
        if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
            throw "aws CLI exited with code $LASTEXITCODE"
        }
        Write-Step $Description 'success'
        return $result
    } catch {
        Write-Step ("$Description -- $($_.Exception.Message)" -replace '--', '-') 'failure'
        return $null
    }
}

function Test-BucketOwnership {
    # Returns $true if the bucket is owned by this account (or does not exist)
    try {
        $null = aws s3api head-bucket --bucket $DocsBucketName --region $AwsRegion 2>&1
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

# ============================================================================
# Section 1  - Banner + prerequisites
# ============================================================================
Write-Host ''
Write-Host '=== TeleGenie AI AWS Deployment ===' -ForegroundColor Magenta
Write-Host ("Region:           {0}" -f $AwsRegion)
Write-Host ("Account ID:       {0}" -f $AwsAccountId)
Write-Host ("Environment:      {0}" -f $EnvironmentName)
Write-Host ("GitHub:           {0}@{1}" -f $GitHubRepo, $GitHubBranch)
Write-Host ("Docs bucket:      {0}" -f $DocsBucketName)
Write-Host ''

if ($WhatIfMode) {
    Write-Step 'WhatIf mode  - no AWS calls will be made' 'warn'
    return
}

Write-Step 'Checking prerequisites'

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Step 'aws CLI not found in PATH. Install: https://aws.amazon.com/cli/' 'failure'
    exit 1
}
Write-Step 'aws CLI found' 'success'

$caller = Invoke-Aws 'Verifying AWS credentials (sts:GetCallerIdentity)' {
    aws sts get-caller-identity --region $AwsRegion --output json | ConvertFrom-Json
}
if (-not $caller) {
    Write-Step 'AWS credentials invalid or missing. Run `aws configure`.' 'failure'
    exit 1
}
Write-Host ("  Account:    {0}" -f $caller.Account)
Write-Host ("  ARN:        {0}" -f $caller.Arn)
Write-Host ("  UserId:     {0}" -f $caller.UserId)
Write-Host ''

# ============================================================================
# Section 2  - S3 bucket for PDFs
# ============================================================================
Write-Step 'Provisioning S3 bucket for PDF uploads' 'info'

if (Test-BucketOwnership) {
    Write-Step "Bucket $DocsBucketName already exists  - skipping create" 'success'
} else {
    $null = Invoke-Aws "Creating S3 bucket $DocsBucketName" {
        aws s3api create-bucket `
            --bucket $DocsBucketName `
            --region $AwsRegion `
            --create-bucket-configuration LocationConstraint=$AwsRegion
    }
}

$null = Invoke-Aws 'Blocking public access on docs bucket' {
    aws s3api put-public-access-block `
        --bucket $DocsBucketName `
        --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
}

$null = Invoke-Aws 'Enabling versioning on docs bucket' {
    aws s3api put-bucket-versioning `
        --bucket $DocsBucketName `
        --versioning-configuration Status=Enabled
}

$tagSpec = 'TagSet=[{Key=Project,Value=telegenie-ai},{Key=Environment,Value=' + $EnvironmentName + '},{Key=Owner,Value=rajubera}]'
$null = Invoke-Aws 'Tagging docs bucket' {
    aws s3api put-bucket-tagging `
        --bucket $DocsBucketName `
        --tagging $tagSpec
}

Write-Host ''

# ============================================================================
# Section 3  - Secrets Manager secrets
# ============================================================================
Write-Step 'Bootstrapping Secrets Manager placeholders' 'info'

foreach ($name in $SecretNames) {
    $created = Invoke-Aws "Creating secret $name" {
        aws secretsmanager create-secret `
            --name $name `
            --description "TeleGenie AI placeholder for $name  - replace via AWS Console" `
            --secret-string "REPLACE_WITH_REAL_VALUE_VIA_AWS_CONSOLE" `
            --region $AwsRegion `
            --output json | ConvertFrom-Json
    }

    # If `aws` exited 0 and we got a JSON ARN back, store it
    if ($created -and $created.ARN) {
        $script:CreatedSecrets += [pscustomobject]@{
            Name = $name
            Arn  = $created.ARN
        }
        continue
    }

    # Fallback: maybe the secret already exists (409). Try describe-secret.
    $existing = Invoke-Aws "Secret $name  - describing existing record" {
        aws secretsmanager describe-secret `
            --secret-id $name `
            --region $AwsRegion `
            --output json | ConvertFrom-Json
    }
    if ($existing -and $existing.ARN) {
        $script:CreatedSecrets += [pscustomobject]@{
            Name = $name
            Arn  = $existing.ARN
        }
    }
}

Write-Host ''

# ============================================================================
# Section 4  - SSM Parameter Store
# ============================================================================
Write-Step 'Bootstrapping SSM Parameter Store entries' 'info'

# Generate a random 32-byte hex value for /telegenie/secret-key if not already set
$existingSecretKey = Invoke-Aws 'Checking for existing /telegenie/secret-key' {
    aws ssm get-parameter --name '/telegenie/secret-key' --with-decryption --region $AwsRegion --output json 2>$null | ConvertFrom-Json
}

if ($existingSecretKey -and $existingSecretKey.Parameter) {
    Write-Step 'Re-using existing /telegenie/secret-key' 'success'
    $secretKeyValue = $existingSecretKey.Parameter.Value
} else {
    $secretKeyValue = & python -c "import secrets; print(secrets.token_hex(32))"
    if (-not $secretKeyValue) {
        # Fallback if python isn't available: use a PowerShell random hex
        $secretKeyValue = -join ((1..64) | ForEach-Object { '{0:x}' -f (Get-Random -Max 16) })
    }
    Write-Step 'Generated fresh 32-byte hex for /telegenie/secret-key' 'success'
}

$null = Invoke-Aws 'Writing /telegenie/secret-key' {
    aws ssm put-parameter `
        --name '/telegenie/secret-key' `
        --value $secretKeyValue `
        --type SecureString `
        --overwrite `
        --region $AwsRegion
}

$null = Invoke-Aws 'Writing /telegenie/aws-access-key-id (placeholder)' {
    aws ssm put-parameter `
        --name '/telegenie/aws-access-key-id' `
        --value 'REPLACE_WITH_AWS_ACCESS_KEY_ID' `
        --type SecureString `
        --overwrite `
        --region $AwsRegion
}

$null = Invoke-Aws 'Writing /telegenie/aws-secret-access-key (placeholder)' {
    aws ssm put-parameter `
        --name '/telegenie/aws-secret-access-key' `
        --value 'REPLACE_WITH_AWS_SECRET_ACCESS_KEY' `
        --type SecureString `
        --overwrite `
        --region $AwsRegion
}

Write-Host ''

# ============================================================================
# Section 5  - ECR repository
# ============================================================================
Write-Step 'Provisioning ECR repository' 'info'

$ecrResult = Invoke-Aws "Creating ECR repo $EcrRepoName" {
    aws ecr create-repository `
        --repository-name $EcrRepoName `
        --image-scanning-configuration scanOnPush=true `
        --region $AwsRegion `
        --output json | ConvertFrom-Json
}

$EcrUri = $null
if ($ecrResult -and $ecrResult.repository -and $ecrResult.repository.repositoryUri) {
    $EcrUri = $ecrResult.repository.repositoryUri
} else {
    # Already exists  - fetch it
    $existingEcr = Invoke-Aws "ECR repo $EcrRepoName  - describing existing" {
        aws ecr describe-repositories `
            --repository-names $EcrRepoName `
            --region $AwsRegion `
            --output json | ConvertFrom-Json
    }
    if ($existingEcr -and $existingEcr.repositories -and $existingEcr.repositories.Count -gt 0) {
        $EcrUri = $existingEcr.repositories[0].repositoryUri
    }
}

if ($EcrUri) {
    Write-Host ("  ECR URI:        {0}" -f $EcrUri)
}

Write-Host ''

# ============================================================================
# Section 6  - IAM role for App Runner (OPTIONAL)
# ============================================================================
if ($CreateCustomApprunnerRole) {
    Write-Step 'Creating custom App Runner instance role' 'info'

    $trustPolicy = @'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "tasks.apprunner.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
'@

    $trustFile = New-TemporaryFile
    Set-Content -Path $trustFile -Value $trustPolicy -Encoding utf8

    $roleResult = Invoke-Aws "Creating IAM role $InstanceRoleName" {
        aws iam create-role `
            --role-name $InstanceRoleName `
            --assume-role-policy-document "file://$($trustFile.FullName)" `
            --output json | ConvertFrom-Json
    }

    if (-not $roleResult) {
        # May already exist
        $existingRole = Invoke-Aws "IAM role $InstanceRoleName  - describing existing" {
            aws iam get-role --role-name $InstanceRoleName --output json | ConvertFrom-Json
        }
    }

    $null = Invoke-Aws 'Attaching AmazonECRReadOnly' {
        aws iam attach-role-policy `
            --role-name $InstanceRoleName `
            --policy-arn arn:aws:iam::aws:policy/AmazonECRReadOnly
    }

    # Single-quoted here-string so PowerShell does NOT try to interpolate $DocsBucketName / $LogGroupName
    $inlinePolicyTemplate = @'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue"],
      "Resource": "arn:aws:secretsmanager:*:*:secret:telegenie/*"
    },
    {
      "Effect": "Allow",
      "Action": ["ssm:GetParameter", "ssm:GetParameters"],
      "Resource": "arn:aws:ssm:*:*:parameter/telegenie/*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
      "Resource": [
        "arn:aws:s3:::__DOCS_BUCKET__",
        "arn:aws:s3:::__DOCS_BUCKET__/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogGroups", "logs:DescribeLogStreams"],
      "Resource": "arn:aws:logs:*:*:log-group:__LOG_GROUP__:*"
    }
  ]
}
'@
    $inlinePolicy = $inlinePolicyTemplate.Replace('__DOCS_BUCKET__', $DocsBucketName).Replace('__LOG_GROUP__', $LogGroupName)

    $policyFile = New-TemporaryFile
    Set-Content -Path $policyFile -Value $inlinePolicy -Encoding utf8

    $null = Invoke-Aws 'Attaching inline secrets/ssm/s3/logs policy' {
        aws iam put-role-policy `
            --role-name $InstanceRoleName `
            --policy-name TelegenieApprunnerAccess `
            --policy-document "file://$($policyFile.FullName)"
    }

    Remove-Item $trustFile, $policyFile -ErrorAction SilentlyContinue
    Write-Host ''
} else {
    Write-Step 'Skipping custom App Runner role (-CreateCustomApprunnerRole not set). App Runner will use its default managed role.' 'warn'
    Write-Host ''
}

# ============================================================================
# Section 7  - CloudWatch Log Group
# ============================================================================
Write-Step 'Provisioning CloudWatch Log Group' 'info'

$null = Invoke-Aws "Creating log group $LogGroupName (30-day retention)" {
    aws logs create-log-group `
        --log-group-name $LogGroupName `
        --region $AwsRegion
}

# If already exists, create-log-group fails but we don't care  - put-retention is idempotent
$null = Invoke-Aws 'Setting log retention to 30 days' {
    aws logs put-retention-policy `
        --log-group-name $LogGroupName `
        --retention-in-days 30
}

Write-Host ''

# ============================================================================
# Section 8  - App Runner service (unless -SkipApprunner)
# ============================================================================
if ($SkipApprunner) {
    Write-Step 'Skipping App Runner service creation (-SkipApprunner set)' 'warn'
} else {
    Write-Step 'Creating App Runner service' 'info'

    # Locate apprunner.yaml
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $candidatePaths = @(
        (Join-Path $scriptDir 'apprunner.yaml'),
        (Join-Path (Get-Location) 'apprunner.yaml')
    )
    $apprunnerYamlPath = $candidatePaths | Where-Object { Test-Path $_ } | Select-Object -First 1

    if (-not $apprunnerYamlPath) {
        Write-Step 'apprunner.yaml not found  - cannot build source configuration' 'failure'
    } else {
        Write-Host ("  Using config:  {0}" -f $apprunnerYamlPath)

        # Render the YAML with the real account ID replacing 000000000000
        $renderedYaml = Get-Content $apprunnerYamlPath -Raw
        $renderedYaml = $renderedYaml -replace '000000000000', $AwsAccountId
        # Also pin the region token if present
        $renderedYaml = $renderedYaml -replace 'us-east-1:000000000000', "$AwsRegion`:$AwsAccountId"

        $tempYaml = New-TemporaryFile
        Set-Content -Path $tempYaml -Value $renderedYaml -Encoding utf8

        # Translate the simplified YAML into the JSON shape that
        # aws apprunner create-service expects.
        $sourceConfig = @{
            AuthenticationConfiguration = @{
                ConnectionArn = ''   # filled below if a GitHub connection ARN is supplied
            }
            AutoDeploymentsEnabled       = $true
            CodeRepository = @{
                RepositoryUrl = "https://github.com/$GitHubRepo"
                SourceCodeVersion = @{
                    Type  = 'BRANCH'
                    Value = $GitHubBranch
                }
                CodeConfiguration = @{
                    ConfigurationSource = 'API'
                    CodeConfigurationValues = @{
                        Runtime = 'PYTHON_3'
                        BuildCommand = 'pip install --upgrade pip && pip install -r requirements.txt'
                        StartCommand = 'streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true'
                        Port = '8501'
                        RuntimeEnvironmentVariables = @(
                            @{ Name = 'TELECOM_RAG_APP_ENV';                Value = $EnvironmentName },
                            @{ Name = 'TELECOM_RAG_AWS_DEFAULT_REGION';     Value = $AwsRegion },
                            @{ Name = 'TELECOM_RAG_AWS_S3_BUCKET';          Value = $DocsBucketName }
                        )
                        # Secrets are also embedded in the YAML  - App Runner resolves ARNs at task start
                        Secrets = @()
                    }
                }
            }
        }

        $sourceConfigJson = $sourceConfig | ConvertTo-Json -Depth 10
        $tempSource = New-TemporaryFile
        Set-Content -Path $tempSource -Value $sourceConfigJson -Encoding utf8

        $instanceConfig = @{
            Cpu               = '1024'
            Memory            = '3072'
            InstanceRoleArn   = ''
        } | ConvertTo-Json

        $healthConfig = @{
            Protocol            = 'HTTP'
            Path                = '/_stcore/health'
            Interval            = 10
            Timeout             = 5
            HealthyThreshold    = 1
            UnhealthyThreshold  = 5
        } | ConvertTo-Json

        $tempInstance = New-TemporaryFile
        $tempHealth   = New-TemporaryFile
        Set-Content -Path $tempInstance -Value $instanceConfig -Encoding utf8
        Set-Content -Path $tempHealth   -Value $healthConfig   -Encoding utf8

        $tagArgs = @(
            'Key=Project,Value=telegenie-ai',
            'Key=Environment,Value=' + $EnvironmentName,
            'Key=Owner,Value=rajubera'
        )

        $createResult = Invoke-Aws 'Calling aws apprunner create-service' {
            aws apprunner create-service `
                --service-name $ServiceName `
                --source-configuration "file://$($tempSource.FullName)" `
                --instance-configuration "file://$($tempInstance.FullName)" `
                --health-check-configuration "file://$($tempHealth.FullName)" `
                --tags $tagArgs `
                --region $AwsRegion `
                --output json | ConvertFrom-Json
        }

        if ($createResult -and $createResult.Service) {
            $script:ServiceArn = $createResult.Service.ServiceArn
            $script:ServiceUrl = $createResult.Service.ServiceUrl
            Write-Host ("  Service ARN:    {0}" -f $script:ServiceArn)
            Write-Host ("  Service URL:    {0}" -f $script:ServiceUrl)
        } else {
            # Service may already exist  - describe it
            $existing = Invoke-Aws 'Listing existing App Runner services' {
                aws apprunner list-services --region $AwsRegion --output json | ConvertFrom-Json
            }
            if ($existing -and $existing.ServiceSummaryList) {
                $match = $existing.ServiceSummaryList | Where-Object { $_.ServiceName -eq $ServiceName } | Select-Object -First 1
                if ($match) {
                    $script:ServiceArn = $match.ServiceArn
                    $described = Invoke-Aws 'Describing existing service' {
                        aws apprunner describe-service --service-arn $match.ServiceArn --region $AwsRegion --output json | ConvertFrom-Json
                    }
                    if ($described -and $described.Service -and $described.Service.ServiceUrl) {
                        $script:ServiceUrl = $described.Service.ServiceUrl
                    }
                    Write-Host ("  Service ARN:    {0}" -f $script:ServiceArn)
                    if ($script:ServiceUrl) {
                        Write-Host ("  Service URL:    {0}" -f $script:ServiceUrl)
                    }
                }
            }
        }

        Remove-Item $tempYaml, $tempSource, $tempInstance, $tempHealth -ErrorAction SilentlyContinue
    }
    Write-Host ''
}

# ============================================================================
# Section 9  - Summary
# ============================================================================
Write-Host '=== Deployment Summary ===' -ForegroundColor Magenta
Write-Host ''
Write-Host ("S3 docs bucket:    {0}" -f $DocsBucketName)
if ($EcrUri) {
    Write-Host ("ECR URI:           {0}" -f $EcrUri)
} else {
    Write-Host 'ECR URI:           <not created>' -ForegroundColor Yellow
}
Write-Host ("Log group:         {0}" -f $LogGroupName)
Write-Host ("App Runner name:   {0}" -f $ServiceName)

if ($ServiceArn) {
    Write-Host ("App Runner ARN:    {0}" -f $ServiceArn)
}
if ($ServiceUrl) {
    Write-Host ("App Runner URL:    https://{0}" -f $ServiceUrl) -ForegroundColor Cyan
} else {
    Write-Host 'App Runner URL:    <pending  - first build takes ~5???8 minutes>' -ForegroundColor Yellow
}

Write-Host ''
Write-Host 'Secrets (paste real values via AWS Console):' -ForegroundColor Yellow
foreach ($s in $CreatedSecrets) {
    Write-Host ("  - {0,-32}  {1}" -f $s.Name, $s.Arn)
}

Write-Host ''
Write-Host 'Next steps:' -ForegroundColor Magenta
Write-Host '  1. AWS Console  - Secrets Manager  - edit each `telegenie/*` secret with the real API key.'
Write-Host '  2. AWS Console  - SSM Parameter Store  - edit `/telegenie/aws-access-key-id` and `/telegenie/aws-secret-access-key` with your IAM access keys.'
Write-Host '  3. Wait for App Runner build to finish (~5???8 min on first deploy).'
Write-Host '  4. Visit the App Runner URL above  - you should see the Streamlit login screen.'
Write-Host '  5. Log in as the bootstrap admin (username printed in the App Runner logs) and CHANGE THE PASSWORD on first login.'
Write-Host ''
Write-Step 'Done.' 'success'