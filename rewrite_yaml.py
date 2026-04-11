import sys
import re

def main():
    with open('.github/workflows/publish-images-to-ecr.yml', 'r') as f:
        content = f.read()

    # We want to find all occurrences of:
    #       - name: Build and push <IMG>
    #         if: ...
    #         uses: docker/build-push-action@v6
    #         with:
    #           context: <CTX>
    #           file: <FILE>
    #           platforms: linux/amd64
    #           push: true
    #           ... (maybe build-args)
    #           tags: ...
    #           cache-from: type=gha
    #           cache-to: type=gha,mode=max
    
    updates = {
        'base-cpu': {'dockerfile': 'images/base/cpu/Dockerfile', 'context': '.', 'repo': 'base-cpu'},
        'base-gpu': {'dockerfile': 'images/base/gpu-torch/Dockerfile', 'context': '.', 'repo': 'base-gpu-torch'},
        'executor-pool': {'dockerfile': 'images/workload/executor-gpu/Dockerfile', 'context': 'images/workload/executor-gpu', 'repo': 'executor-gpu'},
        'executor-pool-cpu': {'dockerfile': 'images/workload/executor-cpu/Dockerfile', 'context': 'images/workload/executor-cpu', 'repo': 'executor-cpu'},
        'ray-worker': {'dockerfile': 'images/framework/ray-gpu/Dockerfile', 'context': '.', 'repo': 'ray-gpu'},
        'training-llm': {'dockerfile': 'images/workload/training-llm/Dockerfile', 'context': '.', 'repo': 'training-llm'},
        'ml-gpu': {'dockerfile': 'images/workload/ml-gpu/Dockerfile', 'context': '.', 'repo': 'ml-gpu'},
        'desk-cpu': {'dockerfile': 'projects/jupyter/images/Dockerfile', 'context': '.', 'repo': 'desk-cpu'},
        'notebook-marimo': {'dockerfile': 'projects/jupyter/images/Dockerfile.marimo', 'context': '.', 'repo': 'notebook-marimo'},
        'desk-gpu': {'dockerfile': 'projects/jupyter/images/Dockerfile.marimo.vscode', 'context': '.', 'repo': 'desk-gpu'},
        'gpu-idle-monitor': {'dockerfile': 'projects/jupyter/gpu-idle-monitor/Dockerfile', 'context': 'projects/jupyter/gpu-idle-monitor', 'repo': 'gpu-idle-monitor'},
        'spark-base': {'dockerfile': 'projects/spark/images/Dockerfile', 'context': '.', 'repo': 'spark-base'},
        'text2sql-serve': {'dockerfile': 'projects/workflows/text2sql/serve/Dockerfile', 'context': 'projects/workflows/text2sql/serve', 'repo': 'text2sql-serve'},
        'workflow-cpu': {'dockerfile': 'projects/workflows/text2sql/images/Dockerfile.cpu', 'context': 'projects/workflows/text2sql', 'repo': 'workflow-cpu'},
        'workflow-gpu': {'dockerfile': 'projects/workflows/text2sql/images/Dockerfile.gpu', 'context': 'projects/workflows/text2sql', 'repo': 'workflow-gpu'},
        'execution-service': {'dockerfile': 'projects/components/services/execution-service/Dockerfile', 'context': 'projects/components/services/execution-service', 'repo': 'execution-service'},
        'registry-service': {'dockerfile': 'projects/components/services/registry-service/Dockerfile', 'context': 'projects/components/services/registry-service', 'repo': 'registry-service'},
        'backend': {'dockerfile': 'projects/backend/Dockerfile', 'context': '.', 'repo': 'backend-api'},
        'dashboard': {'dockerfile': 'projects/dashboard/Dockerfile', 'context': '.', 'repo': 'dashboard'},
    }

    # Regex to match a build block exactly
    pattern = re.compile(
        r"(\s+)- name: Build and push ([^\n]+)\n"
        r"(?:\s+continue-on-error: true\n)?"
        r"(\s+)if: ([^\n]+)\n"
        r"\s+uses: docker/build-push-action@v6\n"
        r"\s+with:\n"
        r".*?"
        r"(?:\s+build-args: \|\n(?P<bargs>(?:\s+.*?\n)+))?"
        r"\s+tags: \|\n(?:\s+.*?\n)+"
        r"\s+cache-from: type=gha\n"
        r"\s+cache-to: type=gha,mode=max\n",
        re.DOTALL
    )

    def replacer(match):
        indent_dash = match.group(1)
        name = match.group(2)
        indent_if = match.group(3)
        condition = match.group(4)
        bargs_raw = match.group('bargs')
        
        upd = updates.get(name)
        if not upd:
            if name == 'backend-api':
                upd = updates['backend']
            else:
                print(f"Unknown image {name}")
                return match.group(0)

        # format args
        bargs = ""
        if bargs_raw:
            args = []
            for line in bargs_raw.split('\n'):
                line = line.strip()
                if line:
                    args.append(f'"{line}"')
            if args:
                bargs = " --build-args " + " ".join(args)
                
        # update condition to include only_images
        new_cond = f"({condition}) || contains(github.event.inputs.only_images, '{upd['repo']}')"
        
        return (
            f"{indent_dash}- name: Build and push {upd['repo']}\n"
            f"{indent_if}if: {new_cond}\n"
            f"{indent_if}run: |\n"
            f"{indent_if}  chmod +x build_image.py\n"
            f"{indent_if}  ./build_image.py --repo {upd['repo']} --dockerfile {upd['dockerfile']} --context {upd['context']} --branch ${{{{ github.event.inputs.source_branch || 'main' }}}} --registry ${{{{ env.ECR_REGISTRY }}}}{bargs}\n"
        )
        
    new_content = pattern.sub(replacer, content)
    
    with open('.github/workflows/publish-images-to-ecr.yml', 'w') as f:
        f.write(new_content)

if __name__ == '__main__':
    main()
