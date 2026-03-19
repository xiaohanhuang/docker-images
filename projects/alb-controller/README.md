# AWS Load Balancer Controller

This component deploys the AWS Load Balancer Controller to manage Application Load Balancers (ALBs) for Kubernetes Ingress resources.

## Prerequisites

- EKS cluster deployed (`projects/eks/`)
- IAM role for service account (IRSA) created via Terraform

## Installation

```bash
# Set the IAM role ARN (exported from Terraform)
export ALB_ROLE_ARN=$(cd ../eks && terraform output -raw alb_controller_role_arn)

# Install the controller
make install

# Or install with explicit role ARN
helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
  --namespace kube-system \
  --version 1.8.1 \
  -f helm-values.yaml \
  --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=$ALB_ROLE_ARN \
  --set clusterName=ml-platform-eks
```

## Verify Installation

```bash
make status

# Check logs
kubectl logs -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller
```

## Usage

Create an Ingress resource with ALB annotations:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: example
  annotations:
    alb.ingress.kubernetes.io/scheme: internal
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:...
spec:
  ingressClassName: alb
  rules:
    - host: example.ml-platform.internal
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: example-service
                port:
                  number: 80
```

## References

- [AWS Load Balancer Controller Documentation](https://kubernetes-sigs.github.io/aws-load-balancer-controller/)
- [Helm Chart](https://github.com/aws/eks-charts/tree/master/stable/aws-load-balancer-controller)
