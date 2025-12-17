import os
import yarp as yp
import pickle
import numpy as np


class YAKS_CLEANUP:
	def __init__(self,feed,dn=1,graph_edit_cutoff=0,dG_mult=1.2,direction='forward',double_ended=True,forward_cutoff=100,hydrated_env=False):
		"""
		This parses the results out of YARP to prepare for microkinetic simulations.  Multiple necessary routines are held here for the YAKS step between reaction exploration and microkinetic simulations.
		It also does post-kinetics decision making.
		The type of processing done depends on the input given: a list for preprocessing, or a cantera object for postprocessing
		"""
		if len(feed) == 6:
			self.path_to_dicts = feed[0]
			self.dG_overall = feed[1]
			self.graph_edit_rp_val = feed[2]
			self.react_yp = feed[3]
			self.prod_yp = feed[4]
			self.dft_theory = feed[5]
		elif len(feed) == 4:
			self.path_to_dicts = feed[0]
			self.net_states = feed[1]
			self.explored_nodes = feed[2]
			self.explored_bimol = feed[3]
		self.dn = dn
		self.graph_edit_cutoff = 0
		self.dG_mult = dG_mult
		self.direction = direction # forward or backward keyword REQUIRED
		self.forward_cutoff = forward_cutoff
		self.hydrated_env = hydrated_env
		self.double_ended = double_ended

	def scrape_yarp(self):
		self.dicts_list = []
		for i in os.listdir(self.path_to_dicts):
			if i.endswith(".p") and not i.startswith("current_summary"):
				self.dicts_list.append(i)

	def load_reactions(self):
		scrape_yarp()
		self.interior_node_list = []
		self.rxn_summaries = {}
		self.node_summaries = {}
		count = -1
		for i in self.dicts_list:
			f = open(f"{self.path_to_dicts}/{i}",'rb')
			rxn_dict = pickle.load(f)
			for rxn in rxn_dict:
				TS_barriers = []
				for conf in rxn.__dict__['rxn'].__dict__['IRC_xtb'].keys():
					if rxn.__dict__['IRC_xtb'][conf]['type'] == 'intended':
						TS_barriers.append(min(rxn.__dict__['rxn'].__dict__['IRC_xtb'][conf]['barriers']))
				if len(TS_barriers) > 0:
					activ_energy = min(TS_barriers)
					reactant_energy = rxn.__dict__['reactant_dft_opt'][self.dft_theory]['thermal']['GibbsFreeEnergy']
					product_energy = rxn.__dict__['product_dft_opt'][self.dft_theory]['thermal']['GibbsFreeEnergy']
					dG_rxn = product_energy - reactant_energy
					if self.direction == 'backward' and dG_rxn < 0:
						if abs(self.dG_overall) >= abs((dG_rxn*dG_mult)) and self.graph_edit_rp_val - graph_edit(react_yp,rxn.__dict__['rxn'].__dict__['product']) >= graph_edit_cutoff:
							react_smi = return_smi_yp(rxn.__dict__['rxn'].__dict__['reactant'])
							prod_smi = return_smi_yp(rxn.__dict__['rxn'].__dict__['product'])
							count += 1
							self.rxn_summaries[count]: {'reactant_smiles':react_smi,'product_smiles':prod_smi,'activation_energy':activ_energy,'dG':dG_rxn}
							if react_smi not in self.interior_node_list:
								self.interior_node_list.append(react_smi)
							#if react_smi not in self.interior_node_list and i.split('.')[0] == 0:  # probably redundant, but left here for ease pre-debug in case it isn't
								#self.interior_node_list.append(react_smi)
							if react_smi not in self.node_summaries.keys():
								self.node_summaries[react_smi] = [prod_smi]
							elif react_smi in self.node_summaries.keys():
								self.node_summaries[react_smi].append(prod_smi)
					elif self.direction == 'backward' and dG_rxn > 0:
						if self.graph_edit_rp_val - graph_edit(react_yp,rxn.__dict__['rxn'].__dict__['product']) >= graph_edit_cutoff:
							react_smi = return_smi_yp(rxn.__dict__['rxn'].__dict__['reactant'])
							prod_smi = return_smi_yp(rxn.__dict__['rxn'].__dict__['product'])
							count += 1
							self.rxn_summaries[count]: {'reactant_smiles':react_smi,'product_smiles':prod_smi,'activation_energy':activ_energy,'dG':dG_rxn}
							if react_smi not in self.interior_node_list:
								self.interior_node_list.append(react_smi)
							#if react_smi not in self.interior_node_list and i.split('.')[0] == 0:
								#self.interior_node_list.append(react_smi)
							if react_smi not in self.node_summaries.keys():
								self.node_summaries[react_smi] = [prod_smi]
							elif react_smi in self.node_summaries.keys():
								self.node_summaries[react_smi].append(prod_smi)
					elif self.direction == 'forward' and activ_energy <= self.forward_cutoff and self.graph_edit_rp_val - graph_edit(prod_yp,rxn.__dict__['rxn'].__dict__['product']) >= graph_edit_cutoff and self.double_ended == True:
						react_smi = return_smi_yp(rxn.__dict__['rxn'].__dict__['reactant'])
						prod_smi = return_smi_yp(rxn.__dict__['rxn'].__dict__['product'])
						count += 1
						self.rxn_summaries[count]: {'reactant_smiles':react_smi,'product_smiles':prod_smi,'activation_energy':activ_energy,'dG':dG_rxn}
					elif self.double_ended == False:
						react_smi = return_smi_yp(rxn.__dict__['rxn'].__dict__['reactant'])
						prod_smi = return_smi_yp(rxn.__dict__['rxn'].__dict__['product'])
						count += 1
						self.rxn_summaries[count]: {'reactant_smiles':react_smi,'product_smiles':prod_smi,'activation_energy':activ_energy,'dG':dG_rxn}
			f.close()
		self.rxn_summaries['interior_nodes'] = self.interior_node_list
		fd = open(f"{self.path_to_dicts}/current_summary.p",'wb')
		pickle.dump(self.rxn_summaries,fd,protocol=pickle.HIGHEST_PROTOCOL)
		fd.close()
		gd = open(f"{self.path_to_dicts}/current_node_summary.p",'wb')
		pickle.dump(self.node_summaries,gd,protocol=pickle.HIGHEST_PROTOCOL)
		gd.close()

	def graph_edit(self,yarpecule1,yarpecule2):
		#get from Ericka + Thomas

	def pick_next_nodes(self,backwards_frac=0.25):
		self.bimol_cands = []
		self.new_nodes = []
		if self.direction == 'forward' or self.double_ended == False:
			max_conc = self.net_states[0][0]
			for val in self.net_states:
				if val[0] > 0.05*max_conc:
					if val[1] in self.explored_nodes:
						self.bimol_cands.append(val[1])
					else:
						self.new_nodes.append(val[1])
			if self.hydrated_env == True and len(self.bimol_cands) > 0:
				for j in self.bimol_cands:
					gate = True
					for k in self.explored_bimol:
						tmp_list = k.split(".")
						if j in tmp_list and "O" in tmp_list:
							gate = False
					if gate == True:
						self.new_nodes.append(f"{j}.O")
			if len(self.bimol_cands) > 1:
				for x,y in enumerate(self.bimol_cands):
					for a,b in enumerate(self.bimol_cands):
						if a > x:
							gate = True
							for k in self.explored_bimol:
								tmp_list = k.split(".")
								if y in tmp_list and b in tmp_list:
									gate = False
							if gate == True:
								self.new_nodes.append(f"{y}.{b}")
		elif self.direction == 'backward':
			outermost_val = []
			outermost_smi = []
			for val in self.net_states:
				if val[1] not in self.explored_nodes:
					outermost_val.append(val[0])
					outermost_smi.append(val[1])
			for i,c in enumerate(outermost_val):
				if c > backwards_frac/len(outermost_val):
					self.new_nodes.append(outermost_smi[i])
			f = open(f"{self.path_to_dicts}/current_node_summary.p",'rb')
			nodes = pickle.load(f)
			for k in nodes.keys():
				accumulator = True
				for smi in nodes[k]:
					if outermost_val[outermost_smi.index(smi)] < (1-backwards_frac)/len(outermost_val):
						accumulator = False
				if accumulator == True:
					self.bimol_cands.append(k)
			if len(self.bimol_cands) > 1:
				for x,y in enumerate(self.bimol_cands):
					for a,b in enumerate(self.bimol_cands):
						if a > x:
							gate = True
							for k in self.explored_bimol:
								tmp_list = k.split(".")
								if y in tmp_list and b in tmp_list:
									gate = False
							if gate == True:
								self.new_nodes.append(f"{y}.{b}")

	# I don't think this function is necessary but I don't want to delete it unless I am certain once this is debugged
	"""
	def check_activity(self):
		if self.direction == 'backward':
			# scan all rxn summaries: use dict keys
			for i in self.rxn_summaries.keys():
				if self.rxn_summaries[i]['product_smiles'] not in self.interior_node_list:
					self.rxn_summaries[i]['activity'] = True
				else:
					self.rxn_summaries[i]['activity'] = False
		elif self.direction == 'forward':
			for i in self.rxn_summaries.keys():
				self.rxn_summaries[i]['activity'] = False
	"""
