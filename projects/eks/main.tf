provider "aws" {
  region  = var.region
  profile = "adfs"
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    # This requires the awscli to be installed locally where Terraform is executed
    args = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--profile", "adfs"]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--profile", "adfs"]
    }
  }
}


data "aws_availability_zones" "available" {}

locals {
  name = var.cluster_name
}

# ──────────────────────────────────────────────
# VPC
# ──────────────────────────────────────────────
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.name}-vpc"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }
}

resource "aws_vpc_endpoint" "s3" {
  vpc_id       = module.vpc.vpc_id
  service_name = "com.amazonaws.${var.region}.s3"
  route_table_ids = concat(
    module.vpc.private_route_table_ids,
    module.vpc.public_route_table_ids
  )
  tags = {
    Name = "${local.name}-s3-endpoint"
  }
}

# ECR interface endpoints — route image pulls over the private AWS network
# instead of the public internet via NAT, cutting pull time from ~15min to ~2min.
resource "aws_security_group" "vpc_endpoints" {
  name        = "${local.name}-vpc-endpoints-sg"
  description = "Allow HTTPS from VPC to interface endpoints"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  tags = {
    Name = "${local.name}-vpc-endpoints-sg"
  }
}

resource "aws_vpc_endpoint" "ecr_dkr" {
  vpc_id              = module.vpc.vpc_id
  service_name        = "com.amazonaws.${var.region}.ecr.dkr"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = module.vpc.private_subnets
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true
  tags = {
    Name = "${local.name}-ecr-dkr-endpoint"
  }
}

resource "aws_vpc_endpoint" "ecr_api" {
  vpc_id              = module.vpc.vpc_id
  service_name        = "com.amazonaws.${var.region}.ecr.api"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = module.vpc.private_subnets
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true
  tags = {
    Name = "${local.name}-ecr-api-endpoint"
  }
}

# ──────────────────────────────────────────────
# EKS Cluster
# ──────────────────────────────────────────────
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = local.name
  cluster_version = "1.34"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true

  enable_cluster_creator_admin_permissions = true

  eks_managed_node_group_defaults = {
    ami_type = "AL2023_x86_64_STANDARD"
  }

  cluster_addons = {
    aws-ebs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_irsa_role.iam_role_arn
    }
    aws-efs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.efs_csi_irsa_role.iam_role_arn
    }
  }

  eks_managed_node_groups = {

    system = {
      name           = "system-group"
      instance_types = ["m5.large"]
      min_size       = 2
      max_size       = 5
      desired_size   = 2
      labels         = { "role" = "system" }
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 100
            volume_type           = "gp3"
            iops                  = 3000
            throughput            = 125
            encrypted             = true
            delete_on_termination = true
          }
        }
      }
    }

    cpu_worker = {
      name           = "cpu-worker-group"
      instance_types = var.cpu_node_instance_types
      min_size       = 0
      max_size       = 10
      desired_size   = 0
      labels         = { "role" = "cpu-worker" }
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 100
            volume_type           = "gp3"
            iops                  = 3000
            throughput            = 125
            encrypted             = true
            delete_on_termination = true
          }
        }
      }
    }

    gpu_worker = {
      name           = "gpu-worker-group"
      ami_type       = "AL2023_x86_64_NVIDIA"
      instance_types = var.gpu_node_instance_types # Default: g5.xlarge (A10G, 24GB VRAM)
      min_size       = 0
      max_size       = 8
      desired_size   = 0
      labels = {
        "role"        = "gpu-worker"
        "accelerator" = "nvidia-a10g"
      }
      taints = {
        dedicated = {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 100
            volume_type           = "gp3"
            iops                  = 3000
            throughput            = 125
            encrypted             = true
            delete_on_termination = true
          }
        }
      }
    }
  }



  access_entries = {
    # One access entry with a policy associated
    karpenter_node = {
      principal_arn = aws_iam_role.karpenter_node.arn
      type          = "EC2_LINUX"
    }
  }
}

module "ebs_csi_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "ebs-csi-role-${local.name}"
  attach_ebs_csi_policy = true

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }
}

module "karpenter_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name                          = "karpenter-controller-${local.name}"
  attach_karpenter_controller_policy = true

  karpenter_controller_cluster_name = module.eks.cluster_name
  karpenter_controller_node_iam_role_arns = [
    aws_iam_role.karpenter_node.arn
  ]

  role_policy_arns = {
    additional = aws_iam_policy.karpenter_controller_additional.arn
  }

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["karpenter:karpenter"]
    }
  }
}

resource "aws_iam_policy" "karpenter_controller_additional" {
  name        = "karpenter-additional-${local.name}"
  description = "Additional permissions for Karpenter controller"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "iam:CreateInstanceProfile",
          "iam:DeleteInstanceProfile",
          "iam:GetInstanceProfile",
          "iam:RemoveRoleFromInstanceProfile",
          "iam:AddRoleToInstanceProfile",
          "iam:TagInstanceProfile",
          "ec2:RunInstances",
          "ec2:CreateFleet",
          "ec2:CreateTags",
          "ec2:TerminateInstances",
          "ec2:DeleteLaunchTemplate"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role" "karpenter_node" {
  name = "karpenter-node-${local.name}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      },
    ]
  })
}

resource "aws_iam_role_policy_attachment" "karpenter_node_eks_worker" {
  role       = aws_iam_role.karpenter_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "karpenter_node_cni" {
  role       = aws_iam_role.karpenter_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "karpenter_node_registry" {
  role       = aws_iam_role.karpenter_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "karpenter_node_ssm" {
  role       = aws_iam_role.karpenter_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "karpenter_node" {
  name = "karpenter-node-${local.name}"
  role = aws_iam_role.karpenter_node.name
}

# ──────────────────────────────────────────────
# ALB Controller IRSA Role
# ──────────────────────────────────────────────
module "alb_controller_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name                              = "alb-controller-${local.name}"
  attach_load_balancer_controller_policy = true

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:aws-load-balancer-controller"]
    }
  }
}

module "s3_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "ml-platform-s3-access-${local.name}"

  role_policy_arns = {
    s3_policy = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
  }

  oidc_providers = {
    ex = {
      provider_arn = module.eks.oidc_provider_arn
      namespace_service_accounts = [
        "jupyter:hub",
        "jupyter:singleuser",
        "flyte:flyte-binary",
        "default:default",
        "ml-platform-development:default",
        "ml-platform-staging:default",
        "ml-platform-production:default",
        "flytesnacks-development:default",
        "flytesnacks-staging:default",
        "flytesnacks-production:default"
      ]
    }
  }
}

output "s3_irsa_role_arn" {
  value = module.s3_irsa_role.iam_role_arn
}

resource "aws_s3_bucket" "this" {
  bucket = "ml-platform-data-${local.name}-${data.aws_caller_identity.current.account_id}"
}

output "s3_bucket_name" {
  value = aws_s3_bucket.this.id
}

data "aws_caller_identity" "current" {}

moved {
  from = module.s3_bucket.aws_s3_bucket.this[0]
  to   = aws_s3_bucket.this
}

resource "kubernetes_storage_class_v1" "gp3" {
  metadata {
    name = "gp3"
    annotations = {
      "storageclass.kubernetes.io/is-default-class" = "true"
    }
  }
  storage_provisioner = "ebs.csi.aws.com"
  reclaim_policy      = "Delete"
  volume_binding_mode = "WaitForFirstConsumer"
  parameters = {
    type      = "gp3"
    encrypted = "true"
  }

  depends_on = [module.eks]
}

# ──────────────────────────────────────────────
# EFS (Elastic File System) for Shared Storage
# ──────────────────────────────────────────────

module "efs_csi_irsa_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "efs-csi-role-${local.name}"
  attach_efs_csi_policy = true

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:efs-csi-controller-sa"]
    }
  }
}

resource "aws_efs_file_system" "shared_data" {
  creation_token = "${local.name}-shared-data"
  encrypted      = true

  tags = {
    Name = "${local.name}-shared-data"
  }
}

resource "aws_security_group" "efs" {
  name        = "${local.name}-efs-sg"
  description = "Allow EFS traffic from VPC"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "EFS from VPC"
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_efs_mount_target" "this" {
  count           = length(module.vpc.private_subnets)
  file_system_id  = aws_efs_file_system.shared_data.id
  subnet_id       = module.vpc.private_subnets[count.index]
  security_groups = [aws_security_group.efs.id]
}

resource "kubernetes_storage_class_v1" "efs" {
  metadata {
    name = "efs-sc"
  }
  storage_provisioner = "efs.csi.aws.com"
  parameters = {
    provisioningMode = "efs-ap"
    fileSystemId     = aws_efs_file_system.shared_data.id
    directoryPerms   = "777"
  }
  mount_options = [
    "iam",
    "tls",
    "lookupcache=none",
    "noatime"
  ]

  depends_on = [module.eks]
}
