import sys
import ruamel.yaml

def main():
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open('.github/workflows/publish-images-to-ecr.yml', 'r') as f:
        y = yaml.load(f)

    # 1. Update on.push.paths
    paths = y['on']['push']['paths']
    for i, p in enumerate(paths):
        if 'projects/components/images' in p:
            paths[i] = p.replace('projects/components/images', 'images')
    if 'images/**' not in paths:
        paths.insert(0, 'images/**')
    y['on']['push']['paths'] = [p for p in paths if not p.startswith('projects/components/images')]

    # 2. Add workflow_dispatch inputs
    if 'workflow_dispatch' not in y['on']:
        y['on']['workflow_dispatch'] = {}
    if y['on']['workflow_dispatch'] is None:
        y['on']['workflow_dispatch'] = {}
    if 'inputs' not in y['on']['workflow_dispatch']:
        y['on']['workflow_dispatch']['inputs'] = {}
        
    y['on']['workflow_dispatch']['inputs']['source_branch'] = {
        'description': 'Branch name to tag images with',
        'type': 'string',
        'default': 'main'
    }
    y['on']['workflow_dispatch']['inputs']['only_images'] = {
        'description': 'Comma separated images to rebuild',
        'type': 'string',
        'default': ''
    }
    if 'force' not in y['on']['workflow_dispatch']['inputs']:
        y['on']['workflow_dispatch']['inputs']['force'] = {
            'description': 'Force rebuild all images',
            'type': 'boolean',
            'default': False
        }

    # 3. Update paths-filter
    filters = """
base_cpu:
  - 'images/base/cpu/**'
  - 'images/versions.env'
base_gpu:
  - 'images/base/gpu-torch/**'
  - 'images/versions.env'
executor_pool:
  - 'images/workload/executor-gpu/**'
  - 'images/base/gpu-torch/**'
  - 'images/versions.env'
executor_pool_cpu:
  - 'images/workload/executor-cpu/**'
  - 'images/base/cpu/**'
  - 'images/versions.env'
ray_worker:
  - 'images/framework/ray-gpu/**'
  - 'images/base/gpu-torch/**'
  - 'images/versions.env'
training_llm:
  - 'images/workload/training-llm/**'
  - 'images/framework/ray-gpu/**'
  - 'images/base/gpu-torch/**'
  - 'images/versions.env'
ml_gpu:
  - 'images/workload/ml-gpu/**'
  - 'images/base/gpu-torch/**'
  - 'images/versions.env'
notebook_cpu:
  - 'projects/jupyter/images/**'
  - 'projects/components/sdk/**'
notebook_marimo:
  - 'projects/jupyter/images/Dockerfile.marimo'
  - 'projects/jupyter/images/requirements.txt'
  - 'projects/components/sdk/**'
notebook_marimo_vscode:
  - 'projects/jupyter/images/Dockerfile.marimo.vscode'
  - 'projects/jupyter/images/vscode_proxy/**'
  - 'projects/jupyter/images/requirements.txt'
  - 'projects/components/sdk/**'
gpu_idle_monitor:
  - 'projects/jupyter/gpu-idle-monitor/**'
spark_base:
  - 'projects/spark/images/**'
text2sql_serve:
  - 'projects/workflows/text2sql/serve/**'
workflow_cpu:
  - 'projects/workflows/text2sql/images/Dockerfile.cpu'
  - 'projects/workflows/text2sql/config.py'
  - 'projects/workflows/text2sql/tasks/**'
  - 'projects/workflows/text2sql/pipeline.py'
workflow_gpu:
  - 'projects/workflows/text2sql/images/Dockerfile.gpu'
  - 'projects/workflows/text2sql/config.py'
  - 'projects/workflows/text2sql/tasks/**'
  - 'projects/workflows/text2sql/pipeline.py'
execution_service:
  - 'projects/components/services/execution-service/**'
registry_service:
  - 'projects/components/services/registry-service/**'
backend:
  - 'projects/backend/**'
  - 'backend/**'
  - 'cli/**'
  - 'pyproject.toml'
dashboard:
  - 'projects/dashboard/**'
"""
    for step in y['jobs']['detect']['steps']:
        if step.get('id') == 'changes':
            step['with']['filters'] = filters.strip()

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

    for job_id, job in y.get('jobs', {}).items():
        if not job_id.startswith('build-'):
            continue
        new_steps = []
        for step in job.get('steps', []):
            if 'uses' in step and 'docker/build-push-action' in step['uses']:
                step_name = step['name']
                img_key = step_name.replace('Build and push ', '')
                upd = updates.get(img_key, updates.get(img_key.replace('-api', '')))
                if not upd: continue
                
                bargs = []
                if 'with' in step and 'build-args' in step['with']:
                    for line in step['with']['build-args'].strip().split('\n'):
                        line = line.strip()
                        if line: bargs.append(line)
                        
                run_cmd = f"chmod +x build_image.py\\n./build_image.py --repo {upd['repo']} --dockerfile {upd['dockerfile']} --context {upd['context']} --branch ${{{{ github.event.inputs.source_branch || 'main' }}}} --registry ${{{{ env.ECR_REGISTRY }}}}"
                if bargs:
                    run_cmd += " --build-args " + " ".join([f'"{b}"' for b in bargs])
                
                cond = step.get('if', '')
                if 'needs.detect.outputs' in cond:
                    cond = f"({cond}) || contains(github.event.inputs.only_images, '{upd['repo']}')"
                    
                new_steps.append({
                    'name': f"Build and push {upd['repo']} (Smart SHA)",
                    'if': cond,
                    'run': run_cmd.replace('\\n', '\n')
                })
            else:
                new_steps.append(step)
        job['steps'] = new_steps

    with open('.github/workflows/publish-images-to-ecr.yml', 'w') as f:
        yaml.dump(y, f)

if __name__ == '__main__':
    main()
