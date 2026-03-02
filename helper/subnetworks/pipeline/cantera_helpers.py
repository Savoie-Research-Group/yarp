"""Core YAML writing and reactor execution helpers for the subnetwork pipeline."""

import cantera as ct
import pandas as pd
import numpy as np

def get_current_species(reaction_dict,depth_dict,curr_depth):
    species = []
    rxns = []
    for i in range(1,curr_depth+1):
        rxns += depth_dict[f"{i}"]
    for i in rxns:
        for j in reaction_dict[f"reaction_{i}"]['reactants']:
            if j not in species:
                species.append(j)
        for j in reaction_dict[f"reaction_{i}"]['products']:
            if j not in species:
                species.append(j)
    return species

def get_current_reactions(reaction_dict,depth_dict,expl_nodes,curr_depth):
    for_rxns = []
    for i in range(1,curr_depth+1):
        for_rxns += depth_dict[f"{i}"]
    rev_rxns = []
    rev_depth = curr_depth - 1
    for i in range(1,rev_depth+1):
        rev_rxns += depth_dict[f"{i}"]
    filtered_rev = []
    for rid in rev_rxns:
        key = f"reaction_{rid}"
        switch = True
        for j in reaction_dict[key]['products']:
            if j not in expl_nodes:
                switch = False
                break
        if switch:
            filtered_rev.append(rid)
    rev_rxns = filtered_rev
    return for_rxns,rev_rxns

def get_rate_constant(rsmi_list,P):
    molecularity = len(rsmi_list)
    K_HAT = 1
    kB = 1.380649*10**-23
    h = 6.62607015*10**-34
    R = 0.000082057366080960
    return (K_HAT*kB/h*(R/P)**(molecularity-1)),(molecularity)

def _element_sort_key(symbol):
    preferred = ["C", "H", "O", "N", "S", "P", "F", "Cl", "Br", "I"]
    if symbol in preferred:
        return (0, preferred.index(symbol))
    return (1, symbol)


def _collect_phase_elements(compound_dict, all_species):
    elements = set()
    for species in all_species:
        composition = compound_dict.get(species, {})
        for symbol, count in composition.items():
            if count:
                elements.add(symbol)
    return sorted(elements, key=_element_sort_key)


def _format_composition(compound_dict, species, phase_elements):
    composition = compound_dict[species]
    parts = []
    for symbol in phase_elements:
        count = composition.get(symbol, 0)
        if count:
            parts.append(f"{symbol}: {count}")
    return ", ".join(parts)


def _yaml_quote(value):
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _canonical_equation(reactants, products):
    left = " + ".join(sorted(str(sp).strip() for sp in reactants if str(sp).strip()))
    right = " + ".join(sorted(str(sp).strip() for sp in products if str(sp).strip()))
    return f"{left} => {right}"


def write_yaml(
    compound_dict,
    reaction_dict,
    depth_dict,
    curr_depth,
    expl_nodes,
    kinetics="gas",
    reactions="all",
    model="constant-cp",
    EOS="ideal-gas",
    T=373,
    P=1,
    initial_species="A",
):
    all_species = get_current_species(reaction_dict,depth_dict,curr_depth)
    initial_species = str(initial_species)
    if initial_species not in all_species:
        raise ValueError(
            f"Initial species token '{initial_species}' is not present in the active species list."
        )
    for_reactions,rev_reactions = get_current_reactions(reaction_dict,depth_dict,expl_nodes,curr_depth)
    phase_elements = _collect_phase_elements(compound_dict, all_species)
    if not phase_elements:
        raise ValueError("No phase elements were found in compound_dict for active species.")
    # Cantera requires explicitly declaring repeated stoichiometric equations as duplicates.
    all_equations = []
    for r in for_reactions:
        ws = f"reaction_{r}"
        reactants = reaction_dict[ws]['reactants']
        products = reaction_dict[ws]['products']
        all_equations.append(_canonical_equation(reactants, products))
    for r in rev_reactions:
        ws = f"reaction_{r}"
        reactants = reaction_dict[ws]['products']
        products = reaction_dict[ws]['reactants']
        all_equations.append(_canonical_equation(reactants, products))
    equation_counts = {}
    for eqn in all_equations:
        equation_counts[eqn] = equation_counts.get(eqn, 0) + 1

    with open("cantera_input.yaml",'w') as ys:
        ys.write("units: {time: s, quantity: mol, activation-energy: kcal/mol, pressure: atm, energy: kcal")
        ys.write("}\n\n")
        ys.write("phases:\n")
        ys.write("- name: Sim\n")
        ys.write(f"  thermo: {EOS}\n")
        ys.write(f"  elements: [{', '.join(phase_elements)}]\n")
        ys.write(f"  kinetics: {kinetics}\n")
        ys.write(f"  reactions: {reactions}\n")
        ys.write("  state: {T: ")
        ys.write(f"{T}, P: {P}, ")
        ys.write("X: {")
        for i, s in enumerate(all_species):
            val = 1 if s == initial_species else 0
            suffix = "," if i < (len(all_species) - 1) else ""
            ys.write(f"{_yaml_quote(s)}: {val}{suffix}")
        ys.write("}")
        ys.write("}\n\n")
        ys.write("species:\n")
        for s in all_species:
            ys.write(f"- name: {_yaml_quote(s)}\n")
            ys.write("  composition: {")
            ys.write(_format_composition(compound_dict, s, phase_elements))
            ys.write("}\n")
            ys.write("  thermo:\n")
            ys.write(f"    model: {model}\n")
            ys.write(f"  equation-of-state: {EOS}\n\n")
        ys.write("reactions:\n")
        for r in for_reactions:
            ws = f"reaction_{r}"
            reactants = reaction_dict[ws]['reactants']
            products = reaction_dict[ws]['products']
            eqn = _canonical_equation(reactants, products)
            ys.write(f'- equation: "{eqn}"\n')
            if equation_counts.get(eqn, 0) > 1:
                ys.write("  duplicate: true\n")
            rc,b = get_rate_constant(reaction_dict[ws]['reactants'],P)
            ys.write("  rate-constant: {A: ")
            ys.write(f"{rc}, b: {b}, Ea: {reaction_dict[ws]['barrier']}")
            ys.write("}\n")
        for r in rev_reactions:
            ws = f"reaction_{r}"
            reactants = reaction_dict[ws]['products']
            products = reaction_dict[ws]['reactants']
            eqn = _canonical_equation(reactants, products)
            ys.write(f'- equation: "{eqn}"\n')
            if equation_counts.get(eqn, 0) > 1:
                ys.write("  duplicate: true\n")
            rc,b = get_rate_constant(reaction_dict[ws]['products'],P)
            ys.write("  rate-constant: {A: ")
            ys.write(f"{rc}, b: {b}, Ea: {reaction_dict[ws]['barrier']-reaction_dict[ws]['dG']}")
            ys.write("}\n")

def _build_isothermal_const_pressure_reactor(sol):
    constructors = []
    if hasattr(ct, "IdealGasConstPressureReactor"):
        constructors.append(ct.IdealGasConstPressureReactor)
    if hasattr(ct, "ConstPressureReactor"):
        constructors.append(ct.ConstPressureReactor)
    if not constructors:
        raise AttributeError("Cantera has no constant-pressure reactor constructor.")

    last_error = None
    for ctor in constructors:
        try:
            return ctor(sol, energy='off', name='isothermal_reactor')
        except TypeError as exc:
            last_error = exc
        try:
            return ctor(sol, energy='off')
        except TypeError as exc:
            last_error = exc
        try:
            return ctor(sol)
        except TypeError as exc:
            last_error = exc

    raise TypeError(
        "Unable to construct an isothermal constant-pressure reactor with this Cantera version."
    ) from last_error


def _reactor_phase(reactor):
    # Cantera >=3.2 uses `phase`; `thermo` is deprecated but retained as fallback.
    phase = getattr(reactor, "phase", None)
    if phase is not None:
        return phase
    return reactor.thermo


def _normalize_terminal_species(terminal_species):
    if terminal_species is None:
        return []
    if isinstance(terminal_species, (list, tuple, set)):
        names = [str(s).strip() for s in terminal_species if str(s).strip()]
    else:
        name = str(terminal_species).strip()
        names = [name] if name else []
    ordered = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _advance_reactor_with_adaptive_extension(
    sim,
    reactor,
    states,
    *,
    terminal_indices,
    terminal_species_names,
    fraction_basis,
    time_step,
    time_sim,
    completion_threshold,
    hold_steps_required,
    use_dxdt_check,
    completion_dxdt_tol,
    min_completion_time=0.0,
    debug_fraction=False,
    extend_if_not_complete=False,
    extend_seconds=None,
    max_time_multiplier=10.0,
):
    phase = _reactor_phase(reactor)
    states.append(phase.state, t=sim.time)
    terminal_indices = list(terminal_indices or [])
    terminal_species_names = list(terminal_species_names or [])
    has_terminal_target = bool(terminal_indices)

    completion_counter = 0
    prev_terminal_value = None
    prev_terminal_time = None
    debug_steps_printed = 0
    reached_completion = False

    target_end_time = float(time_sim)
    base_time = max(float(time_sim), float(time_step))
    extension_chunk_s = float(base_time if extend_seconds is None else extend_seconds)
    if extension_chunk_s <= 0.0:
        extension_chunk_s = base_time
    max_time_s = base_time * max(1.0, float(max_time_multiplier))
    extension_count = 0
    min_completion_time = max(0.0, float(min_completion_time or 0.0))

    while True:
        while sim.time < (target_end_time - 1.0e-15):
            next_time = min(sim.time + time_step, target_end_time)
            sim.advance(next_time)
            phase = _reactor_phase(reactor)
            states.append(phase.state, t=sim.time)
            if not has_terminal_target:
                continue

            terminal_x = float(sum(float(phase.X[idx]) for idx in terminal_indices))
            terminal_y = float(sum(float(phase.Y[idx]) for idx in terminal_indices))
            terminal_value = terminal_x if fraction_basis == "X" else terminal_y
            if debug_fraction and debug_steps_printed < 5:
                parts = []
                for idx, name in zip(terminal_indices, terminal_species_names):
                    parts.append(
                        f"{name}:X={float(phase.X[idx]):.6e},Y={float(phase.Y[idx]):.6e}"
                    )
                print(
                    f"[debug_fraction] t={sim.time:.6g} terminal_state: "
                    f"X={terminal_x:.6e}, Y={terminal_y:.6e} | " + "; ".join(parts)
                )
                debug_steps_printed += 1

            slope_ok = True
            if use_dxdt_check and prev_terminal_value is not None:
                dt_local = float(sim.time - prev_terminal_time)
                if dt_local > 0.0:
                    dterminal_dt = abs((terminal_value - prev_terminal_value) / dt_local)
                else:
                    dterminal_dt = np.inf
                slope_ok = dterminal_dt <= float(completion_dxdt_tol)

            completion_window_open = float(sim.time) >= min_completion_time
            if completion_window_open and terminal_value >= completion_threshold and slope_ok:
                completion_counter += 1
            else:
                completion_counter = 0

            prev_terminal_value = terminal_value
            prev_terminal_time = float(sim.time)
            if completion_counter >= hold_steps_required:
                reached_completion = True
                break

        if reached_completion:
            break
        if not has_terminal_target or not bool(extend_if_not_complete):
            break
        if target_end_time >= (max_time_s - 1.0e-15):
            break

        target_end_time = min(target_end_time + extension_chunk_s, max_time_s)
        extension_count += 1
        if debug_fraction:
            print(
                f"[debug_fraction] extending run window to t={target_end_time:.6g} s "
                f"(max={max_time_s:.6g} s)"
            )

    final_phase = _reactor_phase(reactor)
    final_terminal_value = None
    if has_terminal_target:
        final_terminal_value = float(
            sum(
                float(final_phase.X[idx] if fraction_basis == "X" else final_phase.Y[idx])
                for idx in terminal_indices
            )
        )

    return {
        "completion_reached": bool(reached_completion),
        "final_sim_time_s": float(sim.time),
        "target_end_time_s": float(target_end_time),
        "max_time_s": float(max_time_s),
        "extension_count": int(extension_count),
        "hit_max_time": bool(target_end_time >= (max_time_s - 1.0e-15) and not reached_completion),
        "final_terminal_value": final_terminal_value,
        "min_completion_time_s": float(min_completion_time),
    }


def build_and_run_reactor(input_file,time_sim,time_step,rule,curr_depth,uncertainty=False,uncertainty_cycles=30,scale=3,write_excel=True,terminal_species=None,fraction_basis="X",completion_tol=1e-6,completion_target=None,completion_hold_steps=5,completion_dxdt_tol=0.0,min_completion_time=0.0,debug_fraction=False,extend_if_not_complete=False,extend_seconds=None,max_time_multiplier=10.0,return_details=False):
    if float(time_step) <= 0.0:
        raise ValueError(f"time_step must be > 0, got {time_step}")
    if uncertainty and int(uncertainty_cycles) < 1:
        raise ValueError(f"uncertainty_cycles must be >= 1 when uncertainty=True, got {uncertainty_cycles}")
    sol = ct.Solution(input_file)
    r = _build_isothermal_const_pressure_reactor(sol)
    sim = ct.ReactorNet([r])
    states = ct.SolutionArray(sol, extra=['t'])
    species = states.species_names
    rxns = sol.reactions()
    net_stoich = sol.product_stoich_coeffs - sol.reactant_stoich_coeffs

    fraction_basis = str(fraction_basis).upper()
    if fraction_basis not in {"X", "Y"}:
        raise ValueError(f"fraction_basis must be 'X' or 'Y', got: {fraction_basis}")
    terminal_species_names = _normalize_terminal_species(terminal_species)
    terminal_indices = []
    for name in terminal_species_names:
        if name not in sol.species_names:
            raise ValueError(
                f"terminal_species '{name}' not found. "
                f"Available species: {', '.join(sol.species_names)}"
            )
        terminal_indices.append(sol.species_index(name))
    if completion_target is None:
        completion_threshold = 1.0 - float(completion_tol)
    else:
        completion_threshold = float(completion_target)
    completion_threshold = min(1.0, max(0.0, completion_threshold))
    hold_steps_required = max(1, int(completion_hold_steps))
    use_dxdt_check = float(completion_dxdt_tol) > 0.0

    final_x_all = []
    final_y_all = []
    cnf_all = []
    cum_conc_all = []
    rxn_cum_abs_flux_all = []
    rxn_final_rate_all = []
    species_rxn_prod_flux_all = []
    rxn_cum_abs_flux_ts_all = []
    species_x_ts_all = []
    time_grid_all = []

    extension_diagnostics = []

    if uncertainty:
        tracker, conc, mass_p = [], [], []
        for cycles in range(uncertainty_cycles):
            noise = np.random.normal(0,scale, len(rxns))
            temp = {}
            sim_bar = []
            for i, val in enumerate(sol.reactions()):
                temp = rxns[i].input_data
                temp['rate-constant']['Ea'] = (rxns[i].rate.activation_energy + noise[i])
                sol.modify_reaction(i, ct.Reaction.from_dict(temp, kinetics = sol))
            r = _build_isothermal_const_pressure_reactor(sol)
            sim = ct.ReactorNet([r])
            states = ct.SolutionArray(sol, extra=['t'])
            debug_this_cycle = bool(debug_fraction and cycles == 0)
            extension_info = _advance_reactor_with_adaptive_extension(
                sim,
                r,
                states,
                terminal_indices=terminal_indices,
                terminal_species_names=terminal_species_names,
                fraction_basis=fraction_basis,
                time_step=time_step,
                time_sim=time_sim,
                completion_threshold=completion_threshold,
                hold_steps_required=hold_steps_required,
                use_dxdt_check=use_dxdt_check,
                completion_dxdt_tol=completion_dxdt_tol,
                min_completion_time=min_completion_time,
                debug_fraction=debug_this_cycle,
                extend_if_not_complete=extend_if_not_complete,
                extend_seconds=extend_seconds,
                max_time_multiplier=max_time_multiplier,
            )
            extension_info["cycle_index"] = int(cycles)
            extension_diagnostics.append(extension_info)
            if rule == 'cnf':
                spe_net = np.trapezoid(states.net_production_rates[:], dx=time_step, axis=0)
                net_states = zip(spe_net,species,range(len(species)))
                net_states = sorted(net_states,reverse=True)[:]
                tracker.append(spe_net)
            elif rule == 'css':
                net_states = zip(states.X[-1,:],species,range(len(species)))
                net_states = sorted(net_states,reverse=True)[:]
                tracker.append(states.X[-1,:])
            elif rule == 'final_mass':
                net_states = zip(states.Y[-1,:],species,range(len(species)))
                net_states = sorted(net_states,reverse=True)[:]
                tracker.append(states.Y[-1,:])
            elif rule == 'cum_conc':
                conc_net = np.trapezoid(states.X[:], dx=time_step, axis=0)
                net_states = zip(conc_net,species,range(len(species)))
                net_states = sorted(net_states,reverse=True)[:]
                tracker.append(conc_net)
            mass_p.append(states.Y[-1,:])
            conc.append(states.X[-1,:])

            final_x = np.array(states.X[-1,:], dtype=float)
            final_y = np.array(states.Y[-1,:], dtype=float)
            cnf = np.trapezoid(states.net_production_rates[:], dx=time_step, axis=0)
            cum_conc = np.trapezoid(states.X[:], dx=time_step, axis=0)
            rxn_rop = np.array(states.net_rates_of_progress[:], dtype=float)
            rxn_cum_abs_flux_ts = np.zeros_like(rxn_rop)
            if rxn_rop.shape[0] > 1:
                step_area = 0.5 * (np.abs(rxn_rop[1:]) + np.abs(rxn_rop[:-1])) * time_step
                rxn_cum_abs_flux_ts[1:] = np.cumsum(step_area, axis=0)
            rxn_cum_abs_flux = np.trapezoid(np.abs(rxn_rop), dx=time_step, axis=0)
            rxn_final_rate = np.array(rxn_rop[-1, :], dtype=float)
            species_rxn_prod_flux = np.zeros((len(species), len(rxns)))
            for species_i in range(len(species)):
                for rxn_i in range(len(rxns)):
                    nu = net_stoich[species_i, rxn_i]
                    if nu != 0:
                        contrib = nu * rxn_rop[:, rxn_i]
                        species_rxn_prod_flux[species_i, rxn_i] = np.trapezoid(
                            np.clip(contrib, 0.0, None),
                            dx=time_step,
                        )

            final_x_all.append(final_x)
            final_y_all.append(final_y)
            cnf_all.append(cnf)
            cum_conc_all.append(cum_conc)
            rxn_cum_abs_flux_all.append(rxn_cum_abs_flux)
            rxn_final_rate_all.append(rxn_final_rate)
            species_rxn_prod_flux_all.append(species_rxn_prod_flux)
            rxn_cum_abs_flux_ts_all.append(rxn_cum_abs_flux_ts)
            species_x_ts_all.append(np.array(states.X[:], dtype=float))
            time_grid_all.append(np.array(states.t, dtype=float))

        transposed = zip(*tracker)
        stats = []
        for original_index, (name, values) in enumerate(zip(species,transposed)):
            values = list(values)
            mean = np.mean(values)
            median = np.median(values)
            maximum = np.max(values)
            minimum = np.min(values)
            quartile_25 = np.percentile(values, 25)
            quartile_75 = np.percentile(values, 75)
            num = len(values)
            stats.append({
                'species': name,
                'index': original_index,
                'mean': mean,
                'median': median,
                'max': maximum,
                'min': minimum,
                '25th percentile': quartile_25,
                '75th percentile': quartile_75,
                'number of values': num})
        sorted_by_mean = sorted(stats, key=lambda x: x['mean'], reverse=True)
        sorted_by_median = sorted(stats, key=lambda x: x['median'], reverse=True)
        net_states = [(stat['mean'], stat['species'], stat['index']) for stat in sorted_by_mean]
        if write_excel:
            with pd.ExcelWriter(f"ct_out_{curr_depth}_stats.xlsx") as writer:
                pd.DataFrame(tracker, columns=species).to_excel(writer, sheet_name=rule)
                pd.DataFrame(sim_bar, columns=rxns).to_excel(writer, sheet_name=rule)
                pd.DataFrame(mass_p, columns=species).to_excel(writer, sheet_name='final_mass')
                pd.DataFrame(conc, columns=species).to_excel(writer, sheet_name='final_concentration')
                pd.DataFrame(stats).to_excel(writer, sheet_name='uncertainty_results')
    else:
        extension_info = _advance_reactor_with_adaptive_extension(
            sim,
            r,
            states,
            terminal_indices=terminal_indices,
            terminal_species_names=terminal_species_names,
            fraction_basis=fraction_basis,
            time_step=time_step,
            time_sim=time_sim,
            completion_threshold=completion_threshold,
            hold_steps_required=hold_steps_required,
            use_dxdt_check=use_dxdt_check,
            completion_dxdt_tol=completion_dxdt_tol,
            min_completion_time=min_completion_time,
            debug_fraction=debug_fraction,
            extend_if_not_complete=extend_if_not_complete,
            extend_seconds=extend_seconds,
            max_time_multiplier=max_time_multiplier,
        )
        extension_info["cycle_index"] = 0
        extension_diagnostics.append(extension_info)
        if rule == 'cnf':
            spe_net = np.trapezoid(states.net_production_rates[:], dx=time_step, axis=0)
            net_states = zip(spe_net,species,range(len(species)))
            net_states = sorted(net_states,reverse=True)[:]
        elif rule == 'css':
            net_states = zip(states.X[-1,:],species,range(len(species)))
            net_states = sorted(net_states,reverse=True)[:]
        elif rule == 'final_mass':
            net_states = zip(states.Y[-1,:],species,range(len(species)))
            net_states = sorted(net_states,reverse=True)[:]
        elif rule == 'cum_conc':
            conc_net = np.trapezoid(states.X[:], dx=time_step, axis=0)
            net_states = zip(conc_net,species,range(len(species)))
            net_states = sorted(net_states,reverse=True)[:]
        if write_excel:
            with pd.ExcelWriter(f"ct_out_{curr_depth}_stats.xlsx") as writer:
                if rule == 'cnf':
                    pd.DataFrame([spe_net], columns=species).to_excel(writer,sheet_name='cnf')
                elif rule == 'cum_conc':
                    pd.DataFrame([conc_net], columns=species).to_excel(writer,sheet_name='cum_conc')
                pd.DataFrame(states.X, columns=species).to_excel(writer,sheet_name='concentration')
                pd.DataFrame(states.Y, columns=species).to_excel(writer,sheet_name='mass_percent')
                pd.DataFrame(states.net_rates_of_progress, columns=rxns).to_excel(writer,sheet_name='rxn_fluxes')
                pd.DataFrame(states.net_production_rates, columns=species).to_excel(writer,sheet_name='species_fluxes')

        final_x = np.array(states.X[-1,:], dtype=float)
        final_y = np.array(states.Y[-1,:], dtype=float)
        cnf = np.trapezoid(states.net_production_rates[:], dx=time_step, axis=0)
        cum_conc = np.trapezoid(states.X[:], dx=time_step, axis=0)
        rxn_rop = np.array(states.net_rates_of_progress[:], dtype=float)
        rxn_cum_abs_flux_ts = np.zeros_like(rxn_rop)
        if rxn_rop.shape[0] > 1:
            step_area = 0.5 * (np.abs(rxn_rop[1:]) + np.abs(rxn_rop[:-1])) * time_step
            rxn_cum_abs_flux_ts[1:] = np.cumsum(step_area, axis=0)
        rxn_cum_abs_flux = np.trapezoid(np.abs(rxn_rop), dx=time_step, axis=0)
        rxn_final_rate = np.array(rxn_rop[-1, :], dtype=float)
        species_rxn_prod_flux = np.zeros((len(species), len(rxns)))
        for species_i in range(len(species)):
            for rxn_i in range(len(rxns)):
                nu = net_stoich[species_i, rxn_i]
                if nu != 0:
                    contrib = nu * rxn_rop[:, rxn_i]
                    species_rxn_prod_flux[species_i, rxn_i] = np.trapezoid(
                        np.clip(contrib, 0.0, None),
                        dx=time_step,
                    )

        final_x_all.append(final_x)
        final_y_all.append(final_y)
        cnf_all.append(cnf)
        cum_conc_all.append(cum_conc)
        rxn_cum_abs_flux_all.append(rxn_cum_abs_flux)
        rxn_final_rate_all.append(rxn_final_rate)
        species_rxn_prod_flux_all.append(species_rxn_prod_flux)
        rxn_cum_abs_flux_ts_all.append(rxn_cum_abs_flux_ts)
        species_x_ts_all.append(np.array(states.X[:], dtype=float))
        time_grid_all.append(np.array(states.t, dtype=float))

    final_x_arr = np.array(final_x_all, dtype=float)
    final_y_arr = np.array(final_y_all, dtype=float)
    cnf_arr = np.array(cnf_all, dtype=float)
    cum_conc_arr = np.array(cum_conc_all, dtype=float)
    rxn_cum_abs_flux_arr = np.array(rxn_cum_abs_flux_all, dtype=float)
    rxn_final_rate_arr = np.array(rxn_final_rate_all, dtype=float)
    species_rxn_prod_flux_arr = np.array(species_rxn_prod_flux_all, dtype=float)
    min_len = min(arr.shape[0] for arr in rxn_cum_abs_flux_ts_all)
    rxn_cum_abs_flux_ts_arr = np.array([arr[:min_len] for arr in rxn_cum_abs_flux_ts_all], dtype=float)
    species_x_ts_arr = np.array([arr[:min_len] for arr in species_x_ts_all], dtype=float)
    time_grid_arr = np.array([arr[:min_len] for arr in time_grid_all], dtype=float)
    time_grid_mean = np.mean(time_grid_arr, axis=0)

    details = {
        "species": list(species),
        "reactions": [r.equation for r in rxns],
        "time_grid": time_grid_mean.tolist(),
        "adaptive_extension": {
            "enabled": bool(extend_if_not_complete),
            "extend_seconds": (
                float(time_sim if extend_seconds is None else extend_seconds)
                if bool(extend_if_not_complete)
                else None
            ),
            "max_time_multiplier": float(max_time_multiplier),
            "runs": extension_diagnostics,
        },
        "final_x_mean": final_x_arr.mean(axis=0).tolist(),
        "final_x_std": final_x_arr.std(axis=0).tolist(),
        "species_x_ts_mean": species_x_ts_arr.mean(axis=0).tolist(),
        "species_x_ts_std": species_x_ts_arr.std(axis=0).tolist(),
        "final_y_mean": final_y_arr.mean(axis=0).tolist(),
        "final_y_std": final_y_arr.std(axis=0).tolist(),
        "cnf_mean": cnf_arr.mean(axis=0).tolist(),
        "cnf_std": cnf_arr.std(axis=0).tolist(),
        "cum_conc_mean": cum_conc_arr.mean(axis=0).tolist(),
        "cum_conc_std": cum_conc_arr.std(axis=0).tolist(),
        "rxn_cum_abs_flux_ts_mean": rxn_cum_abs_flux_ts_arr.mean(axis=0).tolist(),
        "rxn_cum_abs_flux_ts_std": rxn_cum_abs_flux_ts_arr.std(axis=0).tolist(),
        "rxn_cum_abs_flux_mean": rxn_cum_abs_flux_arr.mean(axis=0).tolist(),
        "rxn_cum_abs_flux_std": rxn_cum_abs_flux_arr.std(axis=0).tolist(),
        "rxn_final_rate_mean": rxn_final_rate_arr.mean(axis=0).tolist(),
        "rxn_final_rate_std": rxn_final_rate_arr.std(axis=0).tolist(),
        "species_rxn_prod_flux_mean": species_rxn_prod_flux_arr.mean(axis=0).tolist(),
        "species_rxn_prod_flux_std": species_rxn_prod_flux_arr.std(axis=0).tolist(),
    }

    if return_details:
        return net_states,states,details
    return net_states,states
