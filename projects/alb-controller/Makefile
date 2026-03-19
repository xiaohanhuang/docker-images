NAMESPACE := kube-system
CHART_REPO := https://aws.github.io/eks-charts
CHART_NAME := aws-load-balancer-controller
RELEASE_NAME := aws-load-balancer-controller
VERSION := 1.8.1

.PHONY: install upgrade uninstall status deploy-ingress install-all

install:
	@echo "Fetching ALB controller IAM role ARN from Terraform..."
	@cd ../eks && \
		ALB_ROLE_ARN=$$(terraform output -raw alb_controller_role_arn) && \
		CLUSTER_NAME=$$(terraform output -raw cluster_name) && \
		cd ../alb-controller && \
		helm repo add eks $(CHART_REPO) && \
		helm repo update && \
		echo "clusterName: $$CLUSTER_NAME" > /tmp/alb-override-values.yaml && \
		echo "serviceAccount:" >> /tmp/alb-override-values.yaml && \
		echo "  annotations:" >> /tmp/alb-override-values.yaml && \
		echo "    eks.amazonaws.com/role-arn: $$ALB_ROLE_ARN" >> /tmp/alb-override-values.yaml && \
		helm upgrade --install $(RELEASE_NAME) eks/$(CHART_NAME) \
			--namespace $(NAMESPACE) \
			--version $(VERSION) \
			-f helm-values.yaml \
			-f /tmp/alb-override-values.yaml && \
		rm -f /tmp/alb-override-values.yaml

upgrade:
	@cd ../eks && \
		ALB_ROLE_ARN=$$(terraform output -raw alb_controller_role_arn) && \
		CLUSTER_NAME=$$(terraform output -raw cluster_name) && \
		cd ../alb-controller && \
		echo "clusterName: $$CLUSTER_NAME" > /tmp/alb-override-values.yaml && \
		echo "serviceAccount:" >> /tmp/alb-override-values.yaml && \
		echo "  annotations:" >> /tmp/alb-override-values.yaml && \
		echo "    eks.amazonaws.com/role-arn: $$ALB_ROLE_ARN" >> /tmp/alb-override-values.yaml && \
		helm upgrade $(RELEASE_NAME) eks/$(CHART_NAME) \
			--namespace $(NAMESPACE) \
			--version $(VERSION) \
			-f helm-values.yaml \
			-f /tmp/alb-override-values.yaml && \
		rm -f /tmp/alb-override-values.yaml

uninstall:
	helm uninstall $(RELEASE_NAME) --namespace $(NAMESPACE)

status:
	kubectl get pods -n $(NAMESPACE) -l app.kubernetes.io/name=aws-load-balancer-controller

deploy-ingress:
	@echo "Deploying ingress resources..."
	./deploy-ingress.sh

install-all: install deploy-ingress
	@echo "ALB controller and ingress resources installed successfully!"

