#!/usr/bin/env python3
import os
import sys
import json
import hashlib
import subprocess
import argparse

def run_cmd(cmd, check=True, capture=False):
    print(f"Running: {' '.join(cmd)}")
    if capture:
        res = subprocess.run(cmd, check=check, capture_output=True, text=True)
        return res.stdout.strip()
    subprocess.run(cmd, check=check)
    return ""

def hash_files(files):
    h = hashlib.sha256()
    for f in files:
        if os.path.exists(f):
            with open(f, 'rb') as fp:
                h.update(fp.read())
    return h.hexdigest()[:16]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--dockerfile', required=True)
    parser.add_argument('--context', required=True)
    parser.add_argument('--branch', default='main')
    parser.add_argument('--registry', required=True)
    parser.add_argument('--build-args', nargs='*', default=[])
    args = parser.parse_args()

    # Create content hash from Dockerfile + versions.env
    versions_file = "images/versions.env"
    sha_tag = f"sha-{hash_files([args.dockerfile, versions_file])}"
    print(f"Calculated tag: {sha_tag}")

    full_repo = f"ml-platform/{args.repo}"
    
    # Check if this exact SHA already exists in ECR
    check_cmd = [
        "aws", "ecr", "describe-images",
        "--repository-name", full_repo,
        "--image-ids", f"imageTag={sha_tag}",
        "--region", "us-west-2",
        "--output", "json"
    ]
    
    res = subprocess.run(check_cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"✅ Image {full_repo}:{sha_tag} already exists. Skipping build.")
        
        # Get the manifest to retag it
        manifest_cmd = [
            "aws", "ecr", "batch-get-image",
            "--repository-name", full_repo,
            "--image-ids", f"imageTag={sha_tag}",
            "--region", "us-west-2",
            "--query", "images[0].imageManifest",
            "--output", "text"
        ]
        manifest = run_cmd(manifest_cmd, capture=True)
        
        tags_to_apply = []
        if args.branch == 'main':
            tags_to_apply.append('latest')
        else:
            tags_to_apply.append(f"branch-{args.branch}")
            
        for tag in tags_to_apply:
            print(f"Retagging existing image with {tag}...")
            subprocess.run([
                "aws", "ecr", "put-image",
                "--repository-name", full_repo,
                "--image-manifest", manifest,
                "--image-tag", tag,
                "--region", "us-west-2"
            ], capture_output=True)
            
        return 0

    print(f"🔨 Building {full_repo}...")
    
    tags = [f"{args.registry}/{full_repo}:{sha_tag}"]
    if args.branch == 'main':
        tags.append(f"{args.registry}/{full_repo}:latest")
    else:
        tags.append(f"{args.registry}/{full_repo}:branch-{args.branch}")
        
    build_cmd = [
        "docker", "buildx", "build", "--push",
        "--context", args.context,
        "--file", args.dockerfile,
        "--platform", "linux/amd64",
        "--cache-from", "type=gha",
        "--cache-to", "type=gha,mode=max"
    ]
    
    for t in tags:
        build_cmd.extend(["-t", t])
        
    for ba in args.build_args:
        build_cmd.extend(["--build-arg", ba])
        
    run_cmd(build_cmd)
    
if __name__ == "__main__":
    sys.exit(main())
