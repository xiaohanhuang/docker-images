# ──────────────────────────────────────────────
# Route53 Private Hosted Zone for Internal Services
# ──────────────────────────────────────────────
resource "aws_route53_zone" "internal" {
  name = "ml-platform.internal"

  vpc {
    vpc_id = module.vpc.vpc_id
  }

  tags = {
    Name        = "${local.name}-internal-zone"
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

# NOTE: ACM certificate for *.ml-platform.internal is NOT created here.
# ACM DNS validation requires publicly resolvable DNS records, but .internal
# is a private TLD with no public zone possible. The internal ALB uses HTTP
# (port 80) instead of HTTPS. To add HTTPS, register a public domain and
# create a public hosted zone for ACM validation.
