# docker-images

This repository is intended to be a **public** mirror of Docker build contexts from a private monorepo.

It builds and pushes images to Amazon ECR using **GitHub Actions + AWS OIDC** (no long-lived AWS credentials stored in GitHub).

## Required AWS setup

- Create an IAM role that can push to the target ECR repositories.
- Configure the role trust policy to allow `token.actions.githubusercontent.com` for this repo/workflow.

## Required GitHub setup

- Ensure Actions are enabled.
- Update `.github/workflows/publish-images-to-ecr.yml`:
  - `AWS_ROLE_ARN`
  - `ECR_REGISTRY`

## How it stays in sync

The private repo should run a workflow that rsyncs selected directories into this repo (including deletions).

