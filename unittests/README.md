# Tests for xmodmap

## Regression test baselines

Run the unit tests from the repository root with:

```
python -m unittest discover -s unittests -p 'test_*.py'
```


## Metadata

The compressed CSV baselines (`.csv.gz`) contain `__commit__` and
`__worktree_dirty__` metadata. The test helpers also read and write plain CSV
files for compatibility.

The commit identifies Git HEAD when the baseline was generated; a dirty
worktree (including untracked files) means the commit alone does not identify
the exact code used. `unknown` marks older baselines whose provenance was not
recorded. Tests compare numerical results, not the current commit.

After reviewing an intentional behavior change, update an individual baseline
with its test script, for example:

```
python unittests/test_hamiltonian_nonrigid.py --updateTest
```

Updating requires Git and rewrites that test's baseline and metadata.


Compressed baselines can be tracked by Git; uncompressed baseline files are
ignored. Review baseline updates before committing them.
