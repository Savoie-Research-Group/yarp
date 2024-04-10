import yarp as yp
import pickle
import multiprocessing as mp
from multiprocessing import Queue
from logging.handlers import QueueHandler
from joblib import Parallel, delayed
from yarp.find_lewis import all_zeros
from yarp.find_lewis import bmat_unique
import os, sys, yaml, fnmatch
import logging
from openbabel import pybel
from utils import *
from wrappers.reaction import *
from job_mapping import *
from wrappers.crest import CREST
from qc_jobs import *
from conf import *
from analyze_functions import *
from wrappers.pysis import PYSIS
from wrappers.gsm import GSM
from wrappers.reaction import reaction

elements, geo= xyz_parse("reaction_xyz/tst.xyz", multiple=True)
xyz_write(".tmp_R.xyz", elements[0], geo[0])
reactant=yp.yarpecule(".tmp_R.xyz", canon=False)
os.system('rm .tmp_R.xyz')
xyz_write(".tmp_P.xyz", elements[1], geo[1])
product=yp.yarpecule(".tmp_P.xyz", canon=False)
os.system('rm .tmp_P.xyz')
R=reaction(reactant, product, args={}, opt=False)
return_model_rxn(R)
