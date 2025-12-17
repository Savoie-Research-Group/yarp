import yarp as yp
import numpy as np
import yaml
from itertools import combinations
from copy import deepcopy
import sys,os
from yaks_cleanup import *
from helper.cantera_pipeline.cantera_wrapper import *

input_file = "insert_input_file_here"

### process the input file
settings = yaml.load(open(input_file, "r"), Loader=yaml.FullLoader)
keys = [i for i in settings.keys()]
if "direction" not in keys:
	direction = "forward" # default is forward exploration
else:
	direction = settings["direction"]
if "num_kinetics" not in keys:
	num_kinetics = 5
else:
	num_kinetics = settings["num_kinetics"]
if "expl_per_kinetic" not in keys:
	expl_per_kinetic = 5
else:
	expl_per_kinetic = settings["expl_per_kinetic"]
path_to_network = settings["path_to_network"]
double_ended = settings["double_ended"]
if "dG_overall" not in keys:
	dG_overall = None
else:
	dG_overall = settings["dG_overall"]
if "ge_rp_val" not in keys:
	ge_rp_val = None
else:
	ge_rp_val = settings["ge_rp_val"]
if "dft_theory" not in keys:
	dft_theory = "wb97xd/def2-tzvp"
else:
	dft_theory = settings["dft_theory"]
if "Temp" not in keys:
	temperature = 623 # K
else:
	temperature = settings["Temp"]
if "Press" not in keys:
	pressure = 1 # atm
else:
	pressure = settings["Press"]
# get the initial inputs
forward_mols = [i for i in settings["inputs"].split('.')]
sets = []
for i in range(len(forward_mols)):
	sets += list(combinations(x,i+1))
yaks_forward = []
for i in yp_inputs:
	string = ''
	for idx,j in enumerate(i):
		if idx == len(i-1)-1:
			string += j
		else:
			string += j
			string += '.'
	yaks_forward.append(string)
if double_ended == True:
	backward_mols = [i for i in settings["outputs"].split('.')]
	sets = []
	for i in range(len(backward_mols)):
		sets += list(combinations(x,i+1))
	yaks_backward = []
	for i in yp_inputs:
		string = ''
		for idx,j in enumerate(i):
			if idx == len(i-1)-1:
				string += j
			else:
				string += j
				string += '.'
		yaks_backward.append(string)

### initiate lists and add the starting compound to the active loop list
if direction == 'forward':
	active_nodes = deepcopy(yaks_forward)
elif direction == 'backward':
	active_nodes = deepcopy(yaks_backward)
explored_nodes = []
explored_bimol = []
os.mkdir(f"{path_to_network}/{direction}")
os.mkdir(f"{path_to_network}/{direction}/dictionaries")

### shadow initialize loop
for count_kinetics in range(num_kinetics):
	for count_expl in range(expl_per_kinetic):
		# run as many explorations as allowed by the number of active nodes
		if len(active_nodes) == 0:
			pass
		else:

### run reaction analysis on top 5 entries to active loop list- remove from active loop list as completed
			new_input = active_nodes[0]
			os.mkdir(f"{path_to_network}/{direction}/{count_kinetics}_{count_expl}")
			os.mkdir(f"{path_to_network}/{direction}/{count_kinetics}_{count_expl}/input")
			os.mkdir(f"{path_to_network}/{direction}/{count_kinetics}_{count_expl}/output")
			with open(f"{path_to_network}/{direction}/{count_kinetics}_{count_expl}/input/smi.txt") as f:
				f.write(new_input)
			del active_nodes[0]
			if '.' in new_input:
				explored_bimol.append(new_input)
			else:
				explored_nodes.append(new_input)
			submit_yarp(f"{path_to_network}/{direction}/{count_kinetics}_{count_expl}/input/smi.txt")

# compile reaction analysis results
	pre_kinetics = YAKS_CLEANUP([f"{path_to_network}/{direction}/dictionaries",dG_overall,ge_rp_val,yp.yarpecule(settings["inputs"]),yp.yarpecule(settings["outputs"]),dft_theory],direction=direction,double_ended=double_ended)
	pre_kinetics.load_reactions()

# run kinetics
	cantera_obj = CANTERA(input_file,f"{path_to_network}/{direction}/dictionaries",direction=direction,Temperature=temperature,Pressure=pressure)
	cantera_obj.write_yaml()
	cantera_obj.build_and_run_reactor()

# analyze kinetic results
	post_kinetics = YAKS_CLEANUP([f"{path_to_network}/{direction}/dictionaries",cantera_obj.net_states,explored_nodes,explored_bimol],direction=direction,double_ended=double_ended)
	post_kinetics.pick_next_nodes()
	explored_nodes = post_kinetics.explored_nodes
	explored_bimol = post_kinetics.explored_bimol
	active_nodes = post_kinetics.new_nodes

# fact check the active nodes and repeat shadow loop
	print(active_nodes)
	print(explored_nodes)
	print(explored_bimol)
