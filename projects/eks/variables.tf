variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "ml-platform-eks"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "cpu_node_instance_types" {
  description = "Instance types for General Purpose CPU nodes"
  type        = list(string)
  default     = ["m5.xlarge"]
}

variable "gpu_node_instance_types" {
  description = "Instance types for GPU nodes (Training)"
  type        = list(string)
  default     = ["g5.xlarge"] # Using cheaper GPU for dev, switch to p4d in prod
}

