"""Recipe Engine — Declarative ML workflows with infrastructure blueprints.

The Recipe Engine enables users to define ML workflows in YAML with infrastructure
requirements (GPU types, instance counts) and configuration presets. Recipes are
parsed, validated, and translated into Flyte workflows with auto-provisioned
Karpenter NodePools.

Main components:
- schema.py: Pydantic models for recipe validation
- parser.py: YAML parsing and template resolution
- generator.py: Flyte workflow generation (Phase 1+)
- provisioner.py: Karpenter NodePool management (Phase 1+)
- cost.py: Cost estimation (Phase 2+)
"""

__version__ = "0.1.0"

from cli.recipe_engine.schema import Recipe

__all__ = ["Recipe"]
