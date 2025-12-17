import os
import yarp as yp
import pickle
import cantera as ct
import numpy as np
import io
import yaml
import pandas as pd  #Patch used for CSV summaries emitted by the reactor run
from pathlib import Path  #Patch: simple path handling for outputs
from collections import Counter
from cantera_util import elements_from_smiles, split_species

class CANTERA:
	def __init__(self,path_to_settings_file,path_to_dicts,direction='forward',rev_depth=0,rev_depth_hyp=1,state='IdealGas',press="const",therm="isotherm",rule="css",reactions="all",kinetics="gas",model="constant_cp",EOS="ideal-gas",model_reaction=False,parallel=[],uncertainty=False,uncertainty_cycles=50,time_sim=1000,time_step=1,Temperature=298,Pressure=1):
		"""
		This builds and execultes a cantera job file
		Notes on inputs:
		path_to_outfile: transition state analysis output file (should contain barriers and molecular IDs) - assumed .txt
		state: thermodynamics and reactor type
		press: pressure variance
		therm: temperature variance
		rule: analysis rule (cnf,css,final_mass,cum_conc)
		dump: the output file type (.xlsx,.csv,.txt)- .xlsx or .csv are recommended
		reactions: key word to turn off certain reactions (see cantera docs)
		kinetics: matter phase used for simulation
		model_reaction: flag to recognize if model reactions are being used- if True, see "parallel" below for how to chunk
		parallel: list of lists, each of which is a set of model reactions that originate from the same parent molecule (and thus compete for feed material).  If this is empty, all MRs are assumed to originate from the same molecule(s)
		uncertainty: If false, reactions run without uncertainty; otherwise, should be an integer for the radius of the normal distribution sampled for Ea perturbations
		"""
		with open(path_to_settings_file, "r", encoding="utf-8") as fh: self.path_to_settings = yaml.load(fh, Loader=yaml.FullLoader)
		self.path_to_dicts = path_to_dicts
		# load the setings file to check on all the stuff below
		keys = [i for i in self.path_to_settings.keys()]
		self.direction = direction
		self.rev_depth = rev_depth
		self.rev_depth_hyp = rev_depth_hyp
		if "state" not in keys:
			self.state = state
		else:
			self.state = self.path_to_settings["state"]
		if "press" not in keys:
			self.press = press	
		else:
			self.press = self.path_to_settings["press"]
		if "therm" not in keys:
			self.therm = therm
		else:
			self.therm = self.path_to_settings["therm"]
		if "rule" not in keys:
			self.rule = rule
		else:
			self.rule = self.path_to_settings["rule"]
		if "reactions" not in keys:
			self.reactions = reactions
		else:
			self.reactions = self.path_to_settings.get("reactions") or reactions
		if "kinetics" not in keys:
			self.kinetics = kinetics
		else:
			self.kinetics = self.path_to_settings["kinetics"]
		if "model" not in keys:
			self.model = model
		else:
			self.model = self.path_to_settings["model"]
		if "EOS" not in keys:
			self.EOS = EOS
		else:
			self.EOS = self.path_to_settings["EOS"]
		if "model_reaction" not in keys:
			self.model_reaction = model_reaction
		else:
			self.model_reaction = self.path_to_settings["model_reaction"]
		if "parallel" not in keys or self.model_reaction == False:
			self.parallel = parallel
		else:
			self.parallel = self.path_to_settings["parallel"]
		if "uncertainty" not in keys:
			self.uncertainty = uncertainty
		else:
			self.uncertainty = self.path_to_settings["uncertainty"]
		if "uncertainty_cycles" not in keys:
			self.uncertainty_cycles = uncertainty_cycles
		else:
			self.uncertainty_cycles = self.path_to_settings["uncertainty_cycles"]
		if "time_sim" not in keys:
			self.time_sim = time_sim
		else:
			self.time_sim = self.path_to_settings["time_sim"]
		if "time_step" not in keys:
			self.time_step = time_step
		else:
			self.time_step = self.path_to_settings["time_step"]
		if "Temperature" not in keys:
			self.Temperature = Temperature
		else:
			self.Temperature = self.path_to_settings["Temperature"]
		if "Pressure" not in keys:
			self.Pressure = Pressure
		else:
			self.Pressure = self.path_to_settings["Pressure"]

	def get_rate_constant(self,rsmi_list):
		# uses the erying equation, does not automatically pull the imaginary frequency for HAT correction (once that is not recorded by YARP it will be possible)
		molecularity = len(rsmi_list)
		K_HAT = 1
		kB = 1.380649*10**-23
		h = 6.62607015*10**-34
		R = 0.000082057366080960
		return (K_HAT*kB/h*(R/self.Pressure)**(molecularity-1)),(molecularity)


	def pull_initial_species(self):
		self.initial_species = []
		for i in self.path_to_settings["initial_species"]:
			#patch: keep [SMILES, fraction] pairs intact so later indexing works
			if isinstance(i, (list, tuple)):
				if len(i) == 1:
					self.initial_species.append([i[0], 1.0])
				else:
					self.initial_species.append([i[0], i[1]])
			else:
				self.initial_species.append([i, 1.0])

	def _normalize_element_symbol(self, label):
		if not label:
			return label
		label = str(label)
		return label[0].upper() + label[1:].lower()

	def get_elements(self):
		self.elements_list = []
		for i in self.initial_species:
			for element in yp.yarpecule(i[0]).elements:
				symbol = self._normalize_element_symbol(element)
				if symbol not in self.elements_list:
					self.elements_list.append(symbol)

	def open_summary(self):
		self.dicts_list = []
		for i in os.listdir(self.path_to_dicts):
			if i.endswith(".p") and not i.startswith("network_summary.pkl"):
				self.dicts_list.append(i)
		summary_path = Path(self.path_to_dicts) / "network_summary.pkl"
		with open(summary_path, "rb") as fh: self.summary = pickle.load(fh)


	def pull_all_species(self):
		self.all_species = []
		for i in self.summary.keys():
			if i != "interior_nodes":
				if self.summary[i]['reactant_smiles'] not in self.all_species:
					self.all_species.append(self.summary[i]['reactant_smiles'])
				if self.summary[i]['product_smiles'] not in self.all_species:
					self.all_species.append(self.summary[i]['product_smiles'])

	def get_element_dict(self,smi):
		element_dict = {i:0 for i in self.elements_list}
		try:
			comp = elements_from_smiles(smi)
		except Exception:
			comp = {}
		for e, count in comp.items():
			symbol = self._normalize_element_symbol(e)
			if symbol not in element_dict:
				element_dict[symbol] = 0
			element_dict[symbol] += count
		return element_dict

	def write_yaml(self):
		self.open_summary()
		self.pull_initial_species()
		self.get_elements()
		self.pull_all_species()
		self.f = io.StringIO()
		# units are hard coded here; change this if you really want, but this is what is best compatible with YARP
		self.f.write("units: {time: s, quantity: mol, activation-energy: kcal/mol, pressure: atm, energy: kcal")
		self.f.write("}\n\n")
		self.f.write("phases:\n")
		self.f.write("- name: Sim\n")
		self.f.write("  elements: [")
		for i in self.elements_list:
			if i != self.elements_list[-1]:
				self.f.write(f"{i}, ")
			else:
				self.f.write(f"{i}")
		self.f.write("]\n")
		self.f.write(f"  species: [")
		for i in self.all_species:
			if i != self.all_species[-1]:
				self.f.write(f"{i}, ")
			else:
				self.f.write(f"{i}")
		self.f.write("]\n")
		self.f.write(f"  kinetics: {self.kinetics}\n")
		self.f.write(f"  reactions: {self.reactions}\n")
		self.f.write("  state: {T: ")
		self.f.write(f"{self.Temperature}, P: {self.Pressure}, ")
		if self.rule.lower() == "final_mass":
			self.f.write("Y: {")
		else:
			self.f.write("X: {")
		temp_initial = []
		if self.direction == "forward":
			for i in self.initial_species:
				temp_initial.append(i[0])
				if i != self.initial_species[-1]:
					self.f.write(i[0])
					self.f.write(f": {i[1]}, ")
				else:
					self.f.write(i[0])
					self.f.write(f": {i[1]}")
			for i in self.all_species:
				if i not in temp_initial:
					self.f.write(", ")
					self.f.write(f"{i}: {0}")
		elif self.direction == "backward":
			first_species = True
			for i in self.all_species:
				if first_species == True:
					if i not in self.summary['interior_nodes']:
						self.f.write(f",{i}: {1/(len(self.all_species)-len(self.summary['interior_nodes']))}")
					else:
						self.f.write(f",{i}: {0}")
					first_species = False
				else:
					if i not in self.summary['interior_nodes']:
						self.f.write(f",{i}: {1/(len(self.all_species)-len(self.summary['interior_nodes']))}")
					else:
						self.f.write(f",{i}: {0}")	
		self.f.write("}")
		self.f.write("}\n\n")
		self.f.write("species:\n")
		for i in self.all_species:
			self.f.write(f"- name: {i}\n")
			self.f.write(f"  composition: {self.get_element_dict(i)}\n")
			self.f.write("  thermo:\n")
			self.f.write(f"    model: {self.model}\n")
			self.f.write(f"  equation-of-state: {self.EOS}\n\n")
		self.f.write("reactions:\n")
		for key, val in self.summary.items():
			if key == 'interior_nodes':
				continue
			self.f.write(f"- id: '{key}'\n")  #Patch : ensure reactions carry stable identifiers
			if self.direction == 'forward':
				rsmi_list = [s for s in val['reactant_smiles'].split('.') if s]
				psmi_list = [s for s in val['product_smiles'].split('.') if s]
			elif self.direction == 'backward':
				rsmi_list = [s for s in val['product_smiles'].split('.') if s]
				psmi_list = [s for s in val['reactant_smiles'].split('.') if s]
			else:
				rsmi_list = [s for s in val['reactant_smiles'].split('.') if s]
				psmi_list = [s for s in val['product_smiles'].split('.') if s]
			if len(rsmi_list) == 0:
				rsmi_list = ["[H]"]
			if len(psmi_list) == 0:
				psmi_list = ["[H]"]
			r_side = " + ".join(rsmi_list)
			p_side = " + ".join(psmi_list)
			self.f.write(f"  equation: '{r_side} => {p_side}'\n")  # Patch: write actual SMILES equation
			rc,b = self.get_rate_constant(rsmi_list)
			self.f.write("  rate-constant: {A: ")
			if self.direction == 'forward':
				self.f.write(f"{rc}, b: {b}, Ea: {val['activation_energy']}")
			elif self.direction == 'backward':
				self.f.write(f"{rc}, b: {b}, Ea: {val['activation_energy']-val['dG']}")
			else:
				self.f.write(f"{rc}, b: {b}, Ea: {val['activation_energy']}")
			self.f.write("}\n")
	
#	def calc_uncertainty_stats(self):#
		# this is copied directly from Michael Woulfe's original YAKS work- redoing the data collection would require changing this
	    # Transpose the list of lists to group values by their index
#	    transposed = zip(*self.tracker)
#	    
#	    # Initialize a list to hold statistics for each index
#		self.stats = []
#	    
#	    # Calculate statistics for each group of values
#	    for original_index, (name, values) in enumerate(zip(self.species,transposed)):
#	        values = list(values)  # Convert tuple from zip to list for numpy functions
#	        mean = np.mean(values)
#	        median = np.median(values)
#	        maximum = np.max(values)
#	        minimum = np.min(values)
#	        quartile_25 = np.percentile(values, 25)
#	        quartile_75 = np.percentile(values, 75)
#	        num = len(values)
#	        
#	        # Append a dictionary of stats for each index
#	        self.stats.append({
#	            'species': name,
#	            'index': original_index,
#	            'mean': mean,
#	            'median': median,
#	            'max': maximum,
#	            'min': minimum,
#	            '25th percentile': quartile_25,
#	            '75th percentile': quartile_75,
#	            'number of values': num})


	def build_and_run_reactor(self):
		"""
		Patch: reimplemented the reactor driver with Cantera's standard API and CSV outputs.
		"""
		mechanism = self.f.getvalue()
		sol = ct.Solution(yaml=mechanism, phase="gas")
		r = ct.IdealGasConstPressureReactor(
    			sol,
    			energy="off",
    			name="isothermal_reactor",
    			clone=False,   # keep old behavior, silences warning
				)

		sim = ct.ReactorNet([r])
		states = ct.SolutionArray(sol, extra=['t'])
		states.append(r.phase.state, t=sim.time)
		
		while sim.time < self.time_sim:
			sim.advance(sim.time + self.time_step)
			states.append(r.phase.state, t=sim.time)

		self.species = list(states.species_names)
		rxns = sol.reactions()
		rxn_ids = [rx.ID or f"rxn_{idx}" for idx, rx in enumerate(rxns)]

		output_dir = Path(self.path_to_dicts)
		output_dir.mkdir(parents=True, exist_ok=True)

		net_prod = states.net_production_rates
		rxn_rates = states.net_rates_of_progress

		rxn_stats = {
			"max":   np.max(rxn_rates, axis=0),
			"min":   np.min(rxn_rates, axis=0),
			"mean":  np.mean(rxn_rates, axis=0),
			"final": rxn_rates[-1],
		}
		rxn_summary = pd.DataFrame(rxn_stats, index=rxn_ids)
		rxn_summary.index.name = "reaction_id"
		self.reaction_summary_path = output_dir / "reaction_flux_summary.csv"
		rxn_summary.to_csv(self.reaction_summary_path)

		species_stats = {
			"max":   np.max(net_prod, axis=0),
			"min":   np.min(net_prod, axis=0),
			"mean":  np.mean(net_prod, axis=0),
			"final": net_prod[-1],
		}
		species_summary = pd.DataFrame(species_stats, index=self.species)
		species_summary.index.name = "species"
		self.species_summary_path = output_dir / "species_flux_summary.csv"
		species_summary.to_csv(self.species_summary_path)

		final_conc = states.concentrations[-1]
		conc_df = pd.DataFrame([final_conc], columns=self.species)
		self.final_concentration_path = output_dir / "final_concentrations.csv"
		conc_df.to_csv(self.final_concentration_path, index=False)

		final_mole = pd.DataFrame([states.X[-1]], columns=self.species)
		self.final_mole_fraction_path = output_dir / "final_mole_fractions.csv"
		final_mole.to_csv(self.final_mole_fraction_path, index=False)

		# Preserve historical net-state semantics for downstream selection logic
		rule = (self.rule or "css").lower()
		if rule == 'cnf':
			values = np.trapz(net_prod, dx=self.time_step, axis=0)
		elif rule == 'final_mass':
			values = states.Y[-1,:]
		elif rule == 'cum_conc':
			values = np.trapz(states.X, dx=self.time_step, axis=0)
		else:  # css default
			values = states.X[-1,:]
		net_states = list(zip(values, self.species, range(len(self.species))))
		self.net_states = sorted(net_states, key=lambda x: x[0], reverse=True)

		return True


# self variable for the dumps- use these instead of reading the xlsx
			# this is self.net_states
# notes:
