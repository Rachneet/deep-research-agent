# Tests

This directory contains test scripts for the deep research agent.

## Test Files

### `test_individual_nodes.py`
Test individual nodes in complete isolation by calling them as functions.

```bash
python tests/test_individual_nodes.py
```

### `test_minimal_graph.py`
Test nodes within minimal graph configurations to verify routing and state transitions.

```bash
python tests/test_minimal_graph.py
```

## Running Tests

From the project root:

```bash
# Test individual nodes
python tests/test_individual_nodes.py

# Test minimal graph flows
python tests/test_minimal_graph.py

# Or use pytest (if installed)
pytest tests/
```

## Adding New Tests

1. For unit tests of individual nodes: Add to `test_individual_nodes.py`
2. For integration tests of node sequences: Add to `test_minimal_graph.py`
3. For full end-to-end tests: Create separate test files
