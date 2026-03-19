#!/usr/bin/env python3
"""
Test script to validate MLflow PostgreSQL migration.

This script verifies that MLflow can:
1. Connect to PostgreSQL backend
2. Create experiments
3. Log runs, parameters, and metrics
4. Handle concurrent writes

Run this after deploying MLflow with PostgreSQL backend.

Prerequisites:
- kubectl port-forward svc/mlflow -n monitoring 5000:80

Usage:
    python3 mlflow_postgres_smoketest.py
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    print("ERROR: mlflow package not installed. Install with: pip install mlflow")
    sys.exit(1)


def test_basic_connectivity():
    """Test basic connectivity to MLflow server."""
    print("🔍 Testing basic connectivity...")
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        client = MlflowClient()
        # Try to list experiments
        experiments = client.search_experiments()
        print(f"✓ Connected to MLflow server (found {len(experiments)} experiments)")
        return True
    except Exception as e:
        print(f"✗ Failed to connect to MLflow server: {e}")
        return False


def test_create_experiment():
    """Test creating a new experiment."""
    print("🔍 Testing experiment creation...")
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        exp_name = f"postgres-migration-test-{int(time.time())}"
        exp_id = mlflow.create_experiment(exp_name)
        print(f"✓ Created experiment '{exp_name}' (ID: {exp_id})")
        return exp_name
    except Exception as e:
        print(f"✗ Failed to create experiment: {e}")
        return None


def test_log_run(exp_name):
    """Test logging a run with parameters and metrics."""
    print("🔍 Testing run logging...")
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment(exp_name)
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_param("learning_rate", 0.001)
            mlflow.log_param("batch_size", 32)
            mlflow.log_param("optimizer", "adam")

            # Log metrics
            for epoch in range(10):
                mlflow.log_metric("train_loss", 1.0 / (epoch + 1), step=epoch)
                mlflow.log_metric("val_accuracy", epoch * 0.1, step=epoch)

            # Log tags
            mlflow.set_tag("framework", "pytorch")
            mlflow.set_tag("test_type", "postgres_migration")

        print(f"✓ Logged run {run.info.run_id} with params and metrics")
        return run.info.run_id
    except Exception as e:
        print(f"✗ Failed to log run: {e}")
        return None


def test_concurrent_writes(exp_name, num_workers=5):
    """Test concurrent writes to verify no SQLite lock contention."""
    print(f"🔍 Testing concurrent writes ({num_workers} workers)...")

    def create_run(worker_id):
        """Worker function to create a run."""
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment(exp_name)
            with mlflow.start_run():
                mlflow.log_param("worker_id", worker_id)
                mlflow.log_metric("value", worker_id * 0.1)
            return (worker_id, True, None)
        except Exception as e:
            return (worker_id, False, str(e))

    # Run workers in parallel
    successful = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(create_run, i) for i in range(num_workers)]
        for future in as_completed(futures):
            worker_id, success, error = future.result()
            if success:
                successful += 1
            else:
                failed += 1
                print(f"  Worker {worker_id} failed: {error}")

    print(f"✓ Concurrent writes: {successful}/{num_workers} successful, {failed} failed")
    return failed == 0


def test_query_runs(exp_name):
    """Test querying runs from the database."""
    print("🔍 Testing run queries...")
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        client = MlflowClient()
        exp = client.get_experiment_by_name(exp_name)
        if not exp:
            print(f"✗ Experiment '{exp_name}' not found")
            return False
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
        )
        print(f"✓ Found {len(runs)} runs in experiment '{exp_name}'")
        return len(runs) > 0
    except Exception as e:
        print(f"✗ Failed to query runs: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MLflow PostgreSQL Migration Test Suite")
    print("=" * 60)
    print()

    # Test 1: Basic connectivity
    if not test_basic_connectivity():
        print("\n❌ Basic connectivity test failed. Ensure MLflow is running.")
        print("   Run: kubectl port-forward svc/mlflow -n monitoring 5000:80")
        sys.exit(1)

    print()

    # Test 2: Create experiment
    exp_name = test_create_experiment()
    if not exp_name:
        print("\n❌ Experiment creation failed.")
        sys.exit(1)

    print()

    # Test 3: Log run
    run_id = test_log_run(exp_name)
    if not run_id:
        print("\n❌ Run logging failed.")
        sys.exit(1)

    print()

    # Test 4: Concurrent writes (critical for PostgreSQL migration)
    if not test_concurrent_writes(exp_name, num_workers=5):
        print("\n❌ Concurrent writes failed.")
        sys.exit(1)

    print()

    # Test 5: Query runs
    if not test_query_runs(exp_name):
        print("\n❌ Run query failed.")
        sys.exit(1)

    print()
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Verify data in PostgreSQL:")
    print("   kubectl exec -it -n postgres postgres-0 -- psql -U mlflow -d mlflow")
    print("   \\dt  -- List tables")
    print(f"   SELECT * FROM experiments WHERE name='{exp_name}';")
    print()
    print("2. Check MLflow UI:")
    print("   Open http://localhost:5000 in your browser")
    print(f"   Look for experiment: {exp_name}")
    print()


if __name__ == "__main__":
    main()
