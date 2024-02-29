import sys, os
import yarp as yp
from main_xtb import read_rxns, select_rxn_conf
# Read in a reaction geometry from ethene and 1,3-buta-diene to cyclohexane.
from wrappers.reaction import *
logging_queue=""
args={}
args["scratch"]="."
args["scratch_crest"]="./conformer"
args["n_conf"]=10
args["method"]="rdkit"
args["conf_output"]="./rxn_conf"
args["strategy"]=0
args["model_path"]='./bin'
args["scratch_xtb"]='./xtb_run'
args["ff"]="uff"
args["opt"]=False
rxn=[]
rxn.append(read_rxns("reaction_xyz/DA.xyz", args=args))
rxn[0].conf_rdkit()
rxn[0]=rxn[0].rxn_conf_generation(logging_queue)
