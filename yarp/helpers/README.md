# Optional maintenance helpers

`rekey_reactions.py` migrates a **trusted** YARP reaction pickle to the current
reaction hash without changing the input file:

```sh
python -m yarp.helpers.rekey_reactions old.pkl rekeyed.pkl
```

It keeps the first record for each new hash, prints a warning for duplicates,
and writes a separate output pickle. It is not imported or called by normal
reaction generation. Never unpickle data from an untrusted source.
