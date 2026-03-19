output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "region" {
  description = "AWS region"
  value       = var.region
}

output "cluster_name" {
  description = "Kubernetes Cluster Name"
  value       = module.eks.cluster_name
}

output "configure_kubectl" {
  description = "Configure kubectl: Make sure you're logged in with AWS CLI"
  value       = "aws eks --region ${var.region} update-kubeconfig --name ${module.eks.cluster_name}"
}

output "efs_id" {
  description = "EFS file system ID"
  value       = aws_efs_file_system.shared_data.id
}

output "alb_controller_role_arn" {
  description = "IAM role ARN for AWS Load Balancer Controller"
  value       = module.alb_controller_irsa.iam_role_arn
}

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "route53_zone_id" {
  description = "Route53 private hosted zone ID for ml-platform.internal"
  value       = aws_route53_zone.internal.zone_id
}

output "route53_zone_name" {
  description = "Route53 private hosted zone name"
  value       = aws_route53_zone.internal.name
}

# NOTE: ACM certificate not used — .internal TLD cannot be validated via DNS.
# Internal ALB uses HTTP (port 80) instead.
