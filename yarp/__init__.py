# yarp/__init__.py

from rdkit import RDLogger, rdBase
from openbabel import openbabel as ob

# Keep package-wide third-party logger policy in one place. RDKit emits some
# warnings via rdBase rather than the Python RDLogger wrapper, so silence both.
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.info")
rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.info")
ob.obErrorLog.SetOutputLevel(ob.obError)

from yarp.yarpecule.yarpecule import yarpecule
from yarp.yarpecule.lewis.lewis_structure import lewis_struct
