# Sub-Project: EKS

Provisions the foundational AWS EKS cluster, VPC, and GPU node groups.

## ⚠️ IMPORTANT: Configure S3 Remote Backend First

**Before running `terraform apply`, you MUST configure remote state storage.**

Terraform state files contain sensitive information (ARNs, secrets, IP addresses) and should NEVER be committed to git. Configure S3 + DynamoDB backend for secure, collaborative state management.

### Setup Instructions

1. **Create S3 bucket for Terraform state:**
   ```bash
   export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

   aws s3api create-bucket \
     --bucket ml-platform-terraform-state-${AWS_ACCOUNT_ID} \
     --region us-west-2 \
     --create-bucket-configuration LocationConstraint=us-west-2
   ```

2. **Enable versioning and encryption:**
   ```bash
   aws s3api put-bucket-versioning \
     --bucket ml-platform-terraform-state-${AWS_ACCOUNT_ID} \
     --versioning-configuration Status=Enabled

   aws s3api put-bucket-encryption \
     --bucket ml-platform-terraform-state-${AWS_ACCOUNT_ID} \
     --server-side-encryption-configuration '{
       "Rules": [{
         "ApplyServerSideEncryptionByDefault": {
           "SSEAlgorithm": "AES256"
         }
       }]
     }'
   ```

3. **Create DynamoDB table for state locking:**
   ```bash
   aws dynamodb create-table \
     --table-name ml-platform-terraform-locks \
     --attribute-definitions AttributeName=LockID,AttributeType=S \
     --key-schema AttributeName=LockID,KeyType=HASH \
     --billing-mode PAY_PER_REQUEST \
     --region us-west-2
   ```

4. **Configure backend:**
   ```bash
   cp backend.tf.example backend.tf
   # Edit backend.tf and replace <ACCOUNT_ID> with your AWS account ID
   sed -i "s/<ACCOUNT_ID>/${AWS_ACCOUNT_ID}/g" backend.tf
   ```

5. **Initialize Terraform with the new backend:**
   ```bash
   terraform init
   # If migrating from local state: terraform init -migrate-state
   ```

## Deploy

```bash
cd projects/eks
terraform init
terraform apply

# Post-deployment: Install NVIDIA Device Plugin (Required for GPUs)
aws eks --region us-west-2 update-kubeconfig --name ml-platform-eks
kubectl apply -f https://raw.githubusercontent.com/aws/karpenter/v0.32.1/pkg/apis/crds/karpenter.sh_nodeclaims.yaml
kubectl apply -f https://raw.githubusercontent.com/aws/karpenter/v0.32.1/pkg/apis/crds/karpenter.sh_nodepools.yaml
kubectl apply -f https://raw.githubusercontent.com/aws/karpenter/v0.32.1/pkg/apis/crds/karpenter.k8s.aws_ec2nodeclasses.yaml
kubectl apply -f karpenter-nodepool.yaml
kubectl apply -f gp3-sc.yaml
kubectl apply -f nvidia-device-plugin.yaml
kubectl apply -f warm-pool.yaml
```

## Outputs

| Output | Description |
|--------|-------------|
| `cluster_endpoint` | EKS API server URL |
| `cluster_name` | Cluster name for kubectl |
| `configure_kubectl` | Command to set up kubeconfig |

## Dependencies

None — this is the root sub-project. All other sub-projects depend on this.

## Post-Deployment: KEDA

After the EKS cluster is running, install KEDA for inference autoscaling:

```bash
cd projects/keda
make install
make status
```

KEDA works with the `inference-gpu-nodepool` Karpenter NodePool to enable scale-to-zero for GPU inference workloads.
