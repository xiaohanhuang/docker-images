"""Recipe packager for bundling recipes into .mlp archives.

This module provides:
- Bundling recipe YAML, lockfiles, and associated files into .mlp archives
- Archive validation and integrity checks
- Support for hermetic, self-contained recipe distribution
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from cli.recipe_engine.parser import RecipeParser
from cli.recipe_engine.planner import RecipePlanner
from cli.recipe_engine.schema import Recipe

logger = logging.getLogger(__name__)


class RecipePackager:
    """Packager for bundling recipes into .mlp archives."""

    MANIFEST_VERSION = "1.0"
    ARCHIVE_EXTENSION = ".mlp"

    def __init__(self):
        """Initialize the recipe packager."""
        self.parser = RecipeParser()
        self.planner = RecipePlanner()

    def package(
        self,
        recipe_name: str,
        output_path: Optional[Path] = None,
        include_lockfile: bool = True,
        profile: Optional[str] = None,
        component_versions: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Package a recipe into a .mlp archive.

        The archive structure:
        ```
        recipe-name-v1.0.0.mlp/
        ├── manifest.json           # Package metadata
        ├── recipe.yaml             # Recipe definition
        ├── lockfile.yaml           # Component versions (optional)
        ├── workflow.py             # Generated workflow code (optional)
        └── files/                  # Additional files (if any)
        ```

        Args:
            recipe_name: Name of the recipe to package
            output_path: Optional output path for archive (defaults to ./recipe-name-vX.Y.Z.mlp)
            include_lockfile: Whether to include a lockfile
            profile: Infrastructure profile to use for lockfile
            component_versions: Optional component version overrides

        Returns:
            Path to created archive

        Raises:
            FileNotFoundError: If recipe not found
            ValueError: If recipe validation fails
        """
        # Load recipe
        recipe = self.parser.load(recipe_name)

        # Generate archive name
        archive_name = f"{recipe.name}-v{recipe.version}{self.ARCHIVE_EXTENSION}"
        if output_path is None:
            output_path = Path.cwd() / archive_name
        else:
            output_path = Path(output_path)
            if output_path.is_dir():
                output_path = output_path / archive_name

        logger.info(f"Packaging recipe '{recipe.name}' to {output_path}")

        # Create temp directory for staging
        with tempfile.TemporaryDirectory() as tmpdir:
            staging_dir = Path(tmpdir)

            # Write recipe.yaml
            self._write_recipe_yaml(recipe, staging_dir)

            # Generate manifest (write later after workflow code generation)
            manifest = self._generate_manifest(recipe, include_lockfile=include_lockfile)

            # Generate lockfile if requested
            if include_lockfile:
                lockfile = self._generate_lockfile(
                    recipe,
                    profile=profile or self._get_default_profile(recipe),
                    component_versions=component_versions or {},
                )
                self._write_lockfile(lockfile, staging_dir)

            # Generate workflow code (optional)
            try:
                workflow_code = self.planner.generate_workflow_code(
                    recipe, component_versions or {}
                )
                self._write_workflow_code(workflow_code, staging_dir)
                manifest["includes_workflow_code"] = True
            except Exception as e:
                logger.warning(f"Failed to generate workflow code: {e}")
                manifest["includes_workflow_code"] = False

            # Write manifest AFTER workflow code generation so includes_workflow_code is correct
            self._write_manifest(manifest, staging_dir)

            # Copy any additional files (if recipe dir has extra files)
            self._copy_additional_files(recipe_name, staging_dir)

            # Create tarball
            self._create_archive(staging_dir, output_path)

        logger.info(f"Successfully packaged recipe to {output_path}")
        return output_path

    def unpack(
        self,
        archive_path: Path,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Unpack a .mlp archive.

        Args:
            archive_path: Path to .mlp archive
            output_dir: Optional output directory (defaults to ./recipe-name/)

        Returns:
            Path to unpacked recipe directory

        Raises:
            FileNotFoundError: If archive not found
            ValueError: If archive is invalid
        """
        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        # Extract archive
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            with tarfile.open(archive_path, "r:gz") as tar:
                # Security check: ensure no path traversal.
                # Use Path.parts for the ".." check so that legitimate names
                # like "..hidden" are not incorrectly rejected, while catching
                # actual parent-directory components.
                extract_root = tmpdir_path.resolve()
                for member in tar.getmembers():
                    if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                        raise ValueError(f"Invalid archive: unsafe path {member.name}")
                    if member.issym() or member.islnk():
                        raise ValueError(
                            f"Invalid archive: symlink/hardlink not allowed: {member.name}"
                        )
                    # Belt-and-suspenders: reject any member whose resolved path
                    # would escape the extraction root.
                    resolved = (tmpdir_path / member.name).resolve()
                    if not resolved.is_relative_to(extract_root):
                        raise ValueError(
                            f"Invalid archive: path escapes extraction dir: {member.name}"
                        )
                # filter='data' (Python 3.12+) strips special file types and
                # applies additional OS-level hardening automatically.
                tar.extractall(tmpdir_path, filter="data")

            # Read manifest
            manifest_path = tmpdir_path / "manifest.json"
            if not manifest_path.exists():
                raise ValueError("Invalid archive: missing manifest.json")

            with open(manifest_path) as f:
                manifest = json.load(f)

            recipe_name = manifest["recipe"]["name"]

            # Determine output directory
            if output_dir is None:
                output_dir = Path.cwd() / recipe_name
            else:
                output_dir = Path(output_dir)

            # Move unpacked files to output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            for item in tmpdir_path.iterdir():
                target = output_dir / item.name
                if target.exists():
                    logger.warning(f"Overwriting existing file: {target}")
                shutil.move(str(item), str(target))

        logger.info(f"Unpacked recipe to {output_dir}")
        return output_dir

    def validate_archive(self, archive_path: Path) -> Dict[str, Any]:
        """Validate a .mlp archive and return validation results.

        Args:
            archive_path: Path to archive

        Returns:
            Dict with validation results:
            - valid: bool
            - errors: List[str]
            - warnings: List[str]
            - manifest: Dict (if valid)
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "manifest": None,
        }

        archive_path = Path(archive_path)

        # Check file exists
        if not archive_path.exists():
            result["valid"] = False
            result["errors"].append(f"Archive not found: {archive_path}")
            return result

        # Check extension
        if not archive_path.name.endswith(self.ARCHIVE_EXTENSION):
            result["warnings"].append(f"Archive extension should be {self.ARCHIVE_EXTENSION}")

        # Try to extract and validate contents
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                with tarfile.open(archive_path, "r:gz") as tar:
                    # Check for path traversal using Path.parts so that
                    # filenames like "..hidden" are not false-positives.
                    extract_root = tmpdir_path.resolve()
                    for member in tar.getmembers():
                        if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                            result["valid"] = False
                            result["errors"].append(f"Unsafe path in archive: {member.name}")
                            return result
                        if member.issym() or member.islnk():
                            result["valid"] = False
                            result["errors"].append(f"Symlink/hardlink not allowed: {member.name}")
                            return result
                        # Belt-and-suspenders resolved-path check.
                        resolved = (tmpdir_path / member.name).resolve()
                        if not resolved.is_relative_to(extract_root):
                            result["valid"] = False
                            result["errors"].append(f"Path escapes extraction dir: {member.name}")
                            return result

                    # filter='data' (Python 3.12+) adds OS-level hardening.
                    tar.extractall(tmpdir_path, filter="data")

                # Check required files
                manifest_path = tmpdir_path / "manifest.json"
                recipe_path = tmpdir_path / "recipe.yaml"

                if not manifest_path.exists():
                    result["valid"] = False
                    result["errors"].append("Missing manifest.json")

                if not recipe_path.exists():
                    result["valid"] = False
                    result["errors"].append("Missing recipe.yaml")

                if not result["valid"]:
                    return result

                # Load and validate manifest
                with open(manifest_path) as f:
                    manifest = json.load(f)
                    result["manifest"] = manifest

                # Validate manifest structure
                # Note: 'checksum' is intentionally excluded — it is stripped from
                # the manifest before archiving (checksum is of the tarball itself
                # and cannot be embedded inside it).
                required_fields = ["version", "recipe", "created_at"]
                for field in required_fields:
                    if field not in manifest:
                        result["warnings"].append(f"Manifest missing field: {field}")

                # Load and validate recipe
                with open(recipe_path) as f:
                    recipe_data = yaml.safe_load(f)

                try:
                    Recipe(**recipe_data)
                except Exception as e:
                    result["valid"] = False
                    result["errors"].append(f"Recipe validation failed: {e}")

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Failed to validate archive: {e}")

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _write_recipe_yaml(self, recipe: Recipe, staging_dir: Path):
        """Write recipe YAML to staging directory."""
        recipe_path = staging_dir / "recipe.yaml"
        recipe_dict = recipe.model_dump(mode="json")
        with open(recipe_path, "w") as f:
            yaml.dump(recipe_dict, f, default_flow_style=False, sort_keys=False)

    def _generate_manifest(self, recipe: Recipe, include_lockfile: bool = True) -> Dict[str, Any]:
        """Generate manifest metadata."""
        manifest = {
            "version": self.MANIFEST_VERSION,
            "recipe": {
                "name": recipe.name,
                "version": recipe.version,
                "description": recipe.description,
                "author": recipe.author,
                "tags": recipe.tags,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "includes_lockfile": include_lockfile,
            "includes_workflow_code": False,  # Will be updated if generated
            "checksum": None,  # Will be computed after archiving
        }
        return manifest

    def _write_manifest(self, manifest: Dict[str, Any], staging_dir: Path):
        """Write manifest.json to staging directory."""
        manifest_path = staging_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _generate_lockfile(
        self,
        recipe: Recipe,
        profile: str,
        component_versions: Dict[str, str],
    ) -> Dict[str, Any]:
        """Generate lockfile for recipe."""
        lockfile = self.planner.generate_lockfile(recipe, component_versions, profile)
        lockfile["generated_at"] = datetime.now(timezone.utc).isoformat()
        return lockfile

    def _write_lockfile(self, lockfile: Dict[str, Any], staging_dir: Path):
        """Write lockfile.yaml to staging directory."""
        lockfile_path = staging_dir / "lockfile.yaml"
        with open(lockfile_path, "w") as f:
            yaml.dump(lockfile, f, default_flow_style=False, sort_keys=False)

    def _write_workflow_code(self, code: str, staging_dir: Path):
        """Write generated workflow code to staging directory."""
        workflow_path = staging_dir / "workflow.py"
        with open(workflow_path, "w") as f:
            f.write(code)

    def _copy_additional_files(self, recipe_name: str, staging_dir: Path):
        """Copy any additional files from recipe directory."""
        try:
            recipe_dir = self.parser.recipes_dir / recipe_name
            if recipe_dir.is_dir():
                files_dir = staging_dir / "files"
                files_dir.mkdir(exist_ok=True)

                # Copy additional files (exclude recipe.yaml itself)
                for item in recipe_dir.iterdir():
                    if item.name != "recipe.yaml" and item.is_file():
                        target = files_dir / item.name
                        target.write_bytes(item.read_bytes())
        except Exception as e:
            logger.debug(f"No additional files to copy: {e}")

    def _create_archive(self, staging_dir: Path, output_path: Path):
        """Create compressed tarball from staging directory.

        The manifest checksum field cannot be embedded inside the archive
        (chicken-and-egg: checksum is of the tarball itself).  We strip the
        ``checksum`` key before archiving to avoid shipping a misleading
        ``null`` value.  The checksum is logged for external verification.
        """
        # Strip null checksum from manifest so it is not misleading
        manifest_path = staging_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest_data = json.loads(manifest_path.read_text())
                if "checksum" in manifest_data:
                    manifest_data.pop("checksum")
                    manifest_path.write_text(json.dumps(manifest_data, indent=2))
            except Exception as exc:
                logger.debug("Failed to strip checksum from manifest.json: %s", exc)

        with tarfile.open(output_path, "w:gz") as tar:
            for item in staging_dir.iterdir():
                tar.add(item, arcname=item.name)

        # Compute checksum of final archive for external use (e.g., push metadata)
        checksum = self._compute_checksum(output_path)
        logger.info("Archive checksum (sha256): %s", checksum)

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_default_profile(self, recipe: Recipe) -> str:
        """Get default profile for recipe (first profile as defined in YAML)."""
        profiles = list(recipe.infrastructure.profiles.keys())
        return profiles[0] if profiles else "default"

    def list_archive_contents(self, archive_path: Path) -> List[str]:
        """List contents of a .mlp archive.

        Args:
            archive_path: Path to archive

        Returns:
            List of file paths in archive
        """
        archive_path = Path(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        contents = []
        with tarfile.open(archive_path, "r:gz") as tar:
            contents = [member.name for member in tar.getmembers()]

        return contents
