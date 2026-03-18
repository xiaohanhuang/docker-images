#!/bin/bash
# Integration test script for dashboard deployment

set -e

NAMESPACE="ml-platform"
APP_LABEL="app=ml-platform-dashboard"

echo "=== Dashboard Integration Test ==="
echo ""

# 1. Check if namespace exists
echo "1. Checking namespace..."
if kubectl get namespace $NAMESPACE &>/dev/null; then
    echo "✓ Namespace $NAMESPACE exists"
else
    echo "✗ Namespace $NAMESPACE not found"
    exit 1
fi

# 2. Check if deployment exists and is ready
echo ""
echo "2. Checking deployment..."
if kubectl get deployment -n $NAMESPACE ml-platform-dashboard &>/dev/null; then
    echo "✓ Deployment exists"

    READY=$(kubectl get deployment -n $NAMESPACE ml-platform-dashboard -o jsonpath='{.status.readyReplicas}')
    DESIRED=$(kubectl get deployment -n $NAMESPACE ml-platform-dashboard -o jsonpath='{.spec.replicas}')

    if [ "$READY" = "$DESIRED" ]; then
        echo "✓ All replicas ready ($READY/$DESIRED)"
    else
        echo "✗ Not all replicas ready ($READY/$DESIRED)"
        exit 1
    fi
else
    echo "✗ Deployment not found"
    exit 1
fi

# 3. Check if pods are running
echo ""
echo "3. Checking pods..."
POD_COUNT=$(kubectl get pods -n $NAMESPACE -l $APP_LABEL --field-selector=status.phase=Running --no-headers | wc -l)
if [ "$POD_COUNT" -gt 0 ]; then
    echo "✓ $POD_COUNT pod(s) running"
else
    echo "✗ No running pods found"
    exit 1
fi

# 4. Check if service exists
echo ""
echo "4. Checking service..."
if kubectl get service -n $NAMESPACE ml-platform-dashboard &>/dev/null; then
    echo "✓ Service exists"
else
    echo "✗ Service not found"
    exit 1
fi

# 5. Check if ingress exists
echo ""
echo "5. Checking ingress..."
if kubectl get ingress -n $NAMESPACE ml-platform-dashboard &>/dev/null; then
    echo "✓ Ingress exists"

    INGRESS_URL=$(kubectl get ingress -n $NAMESPACE ml-platform-dashboard -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    if [ -n "$INGRESS_URL" ]; then
        echo "✓ Ingress URL: $INGRESS_URL"
    else
        echo "⚠ Ingress URL not yet available (ALB provisioning)"
    fi
else
    echo "✗ Ingress not found"
    exit 1
fi

# 6. Test pod health
echo ""
echo "6. Testing pod health..."
POD_NAME=$(kubectl get pods -n $NAMESPACE -l $APP_LABEL -o jsonpath='{.items[0].metadata.name}')
if kubectl exec -n $NAMESPACE $POD_NAME -- wget -q -O- http://localhost:3000/ &>/dev/null; then
    echo "✓ Dashboard responds to HTTP requests"
else
    echo "✗ Dashboard not responding"
    exit 1
fi

# 7. Check RBAC permissions
echo ""
echo "7. Checking RBAC permissions..."
SA="system:serviceaccount:$NAMESPACE:ml-platform-dashboard"

if kubectl auth can-i list nodes --as=$SA &>/dev/null; then
    echo "✓ Can list nodes (cluster-level)"
else
    echo "✗ Cannot list nodes"
    exit 1
fi

if kubectl auth can-i list pods --as=$SA -n $NAMESPACE &>/dev/null; then
    echo "✓ Can list pods (namespace-level)"
else
    echo "✗ Cannot list pods"
    exit 1
fi

# 8. Check environment variables
echo ""
echo "8. Checking environment variables..."
API_URL=$(kubectl get deployment -n $NAMESPACE ml-platform-dashboard -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="NEXT_PUBLIC_API_URL")].value}')
if [ -n "$API_URL" ]; then
    echo "✓ NEXT_PUBLIC_API_URL configured: $API_URL"
else
    echo "⚠ NEXT_PUBLIC_API_URL not set"
fi

GRAFANA_URL=$(kubectl get deployment -n $NAMESPACE ml-platform-dashboard -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="NEXT_PUBLIC_GRAFANA_URL")].value}')
if [ -n "$GRAFANA_URL" ]; then
    echo "✓ NEXT_PUBLIC_GRAFANA_URL configured: $GRAFANA_URL"
else
    echo "⚠ NEXT_PUBLIC_GRAFANA_URL not set"
fi

# 9. Test backend API connectivity
echo ""
echo "9. Testing backend API connectivity..."
if kubectl get pods -n ml-platform-development -l app=ml-platform-api --field-selector=status.phase=Running &>/dev/null; then
    echo "✓ Backend API pods running"

    if kubectl exec -n $NAMESPACE $POD_NAME -- wget -q -O- http://ml-platform-api.ml-platform-development.svc.cluster.local:8000/health &>/dev/null; then
        echo "✓ Backend API reachable from dashboard"
    else
        echo "⚠ Backend API not reachable (may not be deployed yet)"
    fi
else
    echo "⚠ Backend API not deployed yet"
fi

echo ""
echo "=== All Tests Passed ==="
echo ""
echo "Dashboard is deployed and healthy!"
echo ""
echo "Access the dashboard at:"
if [ -n "$INGRESS_URL" ]; then
    echo "http://$INGRESS_URL"
else
    echo "Port-forward: kubectl port-forward -n $NAMESPACE svc/ml-platform-dashboard 3000:80"
    echo "Then visit: http://localhost:3000"
fi
