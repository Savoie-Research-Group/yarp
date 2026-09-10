# model_scorer Artifacts

Place `poor_model.sav` and `rich_model.sav` in this directory before building
the `erm42/yarp:model_scorer` image.

These files are intentionally ignored by git. The lightweight scorer image owns
the pinned scikit-learn/numpy stack needed to unpickle and run the models; the
base YARP environment does not need scikit-learn for conformer-pair scoring.
