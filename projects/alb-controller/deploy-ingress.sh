#!/bin/bash
set -e

# Deploy ingress resources for internal ALB (HTTP)
# Uses HTTP on port 80 since .internal TLD cannot use ACM (no public DNS for validation)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INGRESS_DIR="${SCRIPT_DIR}/ingress"

echo "Applying ingress resources..."
kubectl apply -f "${INGRESS_DIR}/all-services.yaml"

echo ""
echo "Ingress resources deployed successfully!"
echo ""
echo "Services will be available at:"
echo "  - http://flyte.ml-platform.internal"
echo "  - http://flyteadmin.ml-platform.internal"
echo "  - http://grafana.ml-platform.internal"
echo "  - http://mlflow.ml-platform.internal"
echo "  - http://jupyter.ml-platform.internal"
echo "  - http://prometheus.ml-platform.internal"
echo "  - http://kubecost.ml-platform.internal"
echo "  - http://agent.ml-platform.internal"
echo ""
echo "Note: These URLs are only accessible from within the VPC."
echo "You may need to use a VPN or connect from an EC2 instance."
