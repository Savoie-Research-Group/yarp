# Reaction-hash test utilities

`rekey_reactions.py` is an **offline** migration tool for trusted YARP reaction
pickles. It keeps the first record for each current hash and writes to a new
output file; it is not imported by YARP production code. Example:

```sh
python test/helpers/rekey_reactions.py old.pkl rekeyed.pkl
```

The hand-review drawing generator and PNGs remain local under ignored
`debug/reaction_hash_canonicalization/`; they are not part of this test suite.

`build_reaction_hash_fixtures.py` documents the one-time export of the
prototype stress cases to three portable pickles in `test/pickles/`. It needs
the trusted local prototype corpus under `debug/`, but the tests themselves
do not: `test/yarpecule/test_reaction_hash_corpus.py` loads only the three
committed fixture pickles (200 symmetry, 100 nonisomorphic, 100 direction).
The exact graph oracle in that suite is stereochemistry-blind by design;
stereochemical identity is future yarpecule-hash work.
It also preserves five dropped network-fixture reverse pairs in a fourth
pickle for drawing regeneration and regression checks.

Only unpickle inputs from trusted sources.
