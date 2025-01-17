# pytest folder for different applications/calculators
## Currently, it supports:
  * base (`./`)
  * Zimmerman Grow-String Method (**GSM**) calculator (`GSM/`)
## How to extend
  * Create your own folder here, for example `XTB`, that is for testing using `XTB` as the geometry optimization calculator
  * copy `template.yaml` to the newly created folder
  * modify `template.yaml` in the new folder, add the keyword related to your intended test
    * for example, add `XTB_Calculator: XTB` to use XTB as the calculator
  * In `test_rxn.py`, add your new folder to the `CASES` in `test_file()` function
    * for example, if you want the base, the GSM, and your `XTB` get tested, you should have `CASES = ["", "GSM", "XTB"]`
  * run `pytest -s`
## Issues
  * GSM test requires Intel OneMKL library, need to install it (tested on Negishi@Purdue)
