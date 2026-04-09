import ruamel.yaml
from ruamel.yaml.scalarstring import LiteralScalarString

yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)
yaml.width = 4096  # Prevent wrapping long lines

with open('.github/workflows/publish-images-to-ecr.yml', 'r') as f:
    y = yaml.load(f)

for job_id, job in y.get('jobs', {}).items():
    if not job_id.startswith('build-'):
        continue
    for step in job.get('steps', []):
        if 'run' in step:
            # Clean up the run string, removing escaped newlines that ruamel added
            run_str = step['run'].replace('\\n', '\n').replace('\\ ', ' ')
            step['run'] = LiteralScalarString(run_str)

with open('.github/workflows/publish-images-to-ecr.yml', 'w') as f:
    yaml.dump(y, f)
