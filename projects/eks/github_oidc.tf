module "github_oidc" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-github-oidc-provider"
  version = "~> 5.0"
}

module "github_actions_ecr_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-github-oidc-role"
  version = "~> 5.0"

  name = "github-actions-ecr-push-${local.name}"

  # Trust only the xiaohanhuang/ml-platform repository
  subjects = ["repo:xiaohanhuang/ml-platform:*", "repo:xiaohanhuang/docker-images:*"]

  policies = {
    ECRPowerUser = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
  }
}

output "github_actions_role_arn" {
  description = "IAM Role for GitHub Actions (add to repository secrets as AWS_ROLE_ARN)"
  value       = module.github_actions_ecr_role.arn
}
