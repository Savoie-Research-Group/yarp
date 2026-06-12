import re
import numpy as np

from yarp.properties import el_valence, el_metals, el_expand_octet
from yarp.taffi_functions import return_rings,adjmat_to_adjlist

def smiles2adjmat(smiles):
    """
    In-house Savoie group SMILES parser. Written in python and transparent to debug. The main motivation
    was to consistently handle protonation of radicals and atoms with formal charges. The usual SMILES 
    syntax rules apply, except that square brace annotations are handled specially. Square braces are 
    reserved to annotate the isotope, formal charge, or number of hydrogens that should be added to an
    atom. The isotope number must preceed the element label. The charge and number of hydrogens must be
    after the element label. The formal charge can be specified as +d, ++++, -d, ---, where d is an integer.
    The number of hydrogens to be added can be specified as Hd or HHH, where d is an integer.      

    Parameters
    ----------
    smiles: str
            The smiles string that the user wants to parse.

    Returns
    -------
        # Return tokens, adjmat (with bond orders removed), and atom_info list
    return tokens, np.where(adjmat >= 1, 1, 0).astype(int), atom_info
    tokens: list of str,
            The smiles string is tokenized as part of the parsing. The list of tokens is returned for
            debugging purposes. This list has the [] annotations removed. 

    adjmat: array
            This is numpy array holding the graph defined by the smiles string.

    atom_info: list of tuples
            This list is indexed to the adjacency matrix and contains a tuple for each atom. Each tuple has
            the element token, formal charge, explicit hydrogens, and isotope information as its respective
            elements. 
    """
    
    # When we are ready for production, move all definitions here
    if not hasattr(smiles2adjmat, 'aromatics'):
        smiles2adjmat.aromatics = {"b", "c", "n", "o", "p", "s"}

        # Define the regex patterns for tokenization
        smiles2adjmat.token_pattern = r'(\[[^\]]*\]|[A-Z](?:[a-z]+)?|[a-z]|\d{1}|[=#+\-\\\/.()])' # any token pattern
        smiles2adjmat.atom_pattern = r'([A-Z](?:[a-z]+)?|[a-z])' # used to find the tokens corresponding to atoms

        smiles2adjmat.isotope_pattern = re.compile(r'^\[(\d+)') # used to find isotopic information
        smiles2adjmat.charge_pattern = re.compile(r'(\+\d+|-\d+|\++(?!\d)|-+(?!\d))') # used to find explicit partial charge information
        smiles2adjmat.hydrogen_pattern = re.compile(r'(H\d+|H+)') # used to find the explicit specification of hydrogens
        smiles2adjmat.element_label_pattern = re.compile(r'([A-GI-Z](?:[a-gi-z]+)?|[a-gi-z])') # matches labels inside square braces (less h/H)

        
    # Find all matches of the pattern in the SMILES string for tokenization
    preliminary_tokens = re.findall(smiles2adjmat.token_pattern, smiles)
    
    # Initialize the atom_info, tokens lists
    atom_info = [] # holds element label, formal charge, isotope, explicit h information
    tokens = [] # holds the tokens with the [] information removed.
    
    # Regular expressions for parsing atom information
 
    # This loop handles [] annotations and initialization of atom_info list 
    for token in preliminary_tokens:

        if token.startswith('['):
            # Default values
            formal_charge = 0
            explicit_hydrogens = 0 # SMILES convention is no hydrogen inference is [] annotation is being used. 
            isotope = None

            # Check for isotope
            isotope_match = smiles2adjmat.isotope_pattern.search(token)
            if isotope_match:
                isotope = int(isotope_match.group(1))

            # Check for formal charge
            charge_match = smiles2adjmat.charge_pattern.search(token)
            if charge_match:
                charge_str = charge_match.group(1)             
                if charge_str[-1].isdigit():
                    # Case: +2, -3, etc.
                    formal_charge = int(charge_str)
                else:
                    # Case: +, -, ++, ---
                    formal_charge = charge_str.count('+') - charge_str.count('-')

            # Check for explicit hydrogens
            hydrogen_match = smiles2adjmat.hydrogen_pattern.search(token)          
            if hydrogen_match:
                h_str = hydrogen_match.group(1)
                if h_str[-1].isdigit():
                    # Case: H2, H3, etc.
                    explicit_hydrogens = int(h_str[1:])
                elif h_str == ']':
                    # Case: H (only one hydrogen)
                    explicit_hydrogens = 1
                else:
                    # Case: HH, HHH, etc.
                    explicit_hydrogens = len(h_str)

            # Append the atom information to token_info (element,formal,n_hydrogens,isotope)
            # if/else handles case where the atom label is undetected
            element_label_match = smiles2adjmat.element_label_pattern.search(token)
            if element_label_match:
                atom_info.append([element_label_match.group(1), formal_charge, explicit_hydrogens, isotope])
                # Append the clean token information
                tokens.append(element_label_match.group(1))
            else:
                print(f"Error: Could not detect element label at token {token}.")
                return None, None
            
        # Default values                
        elif re.match(smiles2adjmat.atom_pattern, token):

            formal_charge = 0
            explicit_hydrogens = None # None will trigger hydrogen inference in add_hydrogens
            isotope = None
            atom_info.append([token, formal_charge, explicit_hydrogens, isotope])
            tokens.append(token)
        else:
            tokens.append(token)

    # Find all atoms in the SMILES string less the square bracket information (dodges explicit hydrogen issues)
    atoms = re.findall(smiles2adjmat.atom_pattern, ''.join(tokens))
    
    # Initialize adjmat as a square matrix with a row and column for each atom
    adjmat = np.zeros([len(atoms),len(atoms)])

    # Track the indices of atoms in the token list and their branch levels
    atom_indices = []
    branch_levels = []
    current_level = 0    
    branch_open_atom_indices = [] 
    branch_close_atom_indices = [] 
    open_flag = False 
    close_flag = False 
    ring_numbers = {}  # Dictionary to track the atoms preceding each ring number

    # Iterate through the tokens and collect the atom indices and branch levels
    # while we are at it we will also add the bonds associated with rings. 
    for i, token in enumerate(tokens):
        if token == '(':
            current_level += 1
            open_flag = True
        elif token == ')':
            current_level -= 1
            close_flag = True
        elif re.match(smiles2adjmat.atom_pattern, token):
            atom_indices.append(i)
            branch_levels.append(current_level)     
            if close_flag and not open_flag:
                close_flag = False
                branch_close_atom_indices.append(len(atom_indices)-1)
            if open_flag:
                open_flag = False
                branch_open_atom_indices.append(len(atom_indices)-1)
        elif token.isdigit():
            atom_index = len(atom_indices)-1  # The index of the atom preceding the number

            # If the number has been seen before, form a bond between the current atom and the one previously noted
            if token in ring_numbers:
                
                # Note the current atom's index and bond type
                if tokens[i-1] == '-':
                    last_bond_type = 1
                elif tokens[i-1] == '=':
                    last_bond_type = 2
                elif tokens[i-1] == '#':
                    last_bond_type = 3
                else:
                    last_bond_type = None                    
                
                prev_atom_index, prev_bond_type = ring_numbers[token]
                bond_type = last_bond_type if last_bond_type else prev_bond_type
                if prev_bond_type and last_bond_type and prev_bond_type != last_bond_type:
                    print(f"Error: Inconsistent bond order specified for ring closure at token {token}.")
                    return None, None
                
                # If neither closure specified a bond_type then it will be none here and should to default to 1.
                if bond_type is None:
                    bond_type = 1

                adjmat[atom_index, prev_atom_index] = adjmat[prev_atom_index, atom_index] = bond_type
                del ring_numbers[token]  # Allow ring tokens to be reused
            else:
                # Note the current atom's index and bond type
                if tokens[i-1] == '-':
                    last_bond_type = 1
                elif tokens[i-1] == '=':
                    last_bond_type = 2
                elif tokens[i-1] == '#':
                    last_bond_type = 3
                else:
                    last_bond_type = None                    
                ring_numbers[token] = (atom_index, last_bond_type)

    # Iterate through the atom indices and set bonding information for sequential atoms
    for i in range(len(atom_indices)-1):
        
        # Current and next atom indices in the tokens list
        current_atom_index = atom_indices[i]
        next_atom_index = atom_indices[i + 1]
        
        # Extract intervening tokens between current and next atom
        intervening_tokens = tokens[current_atom_index + 1: next_atom_index]
        
        # Check if intervening tokens contain any special characters that prohibit bond formation
        if not any(re.match(r'[\[\].()]',token) for token in intervening_tokens):

            after_digits = []
            for j in intervening_tokens[::-1]:
                if j.isdigit():
                    break
                else:
                    after_digits.append(j)
            
            # Check for double bond (=)
            if '=' in after_digits:
                adjmat[i, i + 1] = adjmat[i + 1, i] = 2
            # Check for triple bond (#)
            elif '#' in after_digits:
                adjmat[i, i + 1] = adjmat[i + 1, i] = 3
            # Update the adjacency matrix for a single bond                
            else:
                adjmat[i, i + 1] = adjmat[i + 1, i] = 1

    # Iterate through the branch starting points and add bonds. 
    for i in branch_open_atom_indices:

        # Current and next atom indices in the tokens list
        current_atom_index = atom_indices[i]
        current_branch_level = branch_levels[i]
        
        # If the user puts an opening parentheses then no bond is required 
        if current_atom_index == 0:
            continue
        else:
            
            # Look backwards until finding an atom at lower branch level
            for j in range(i-1,-1,-1):

                if branch_levels[j] == current_branch_level - 1:

                    # Extract intervening tokens between the branch start and the current atom
                    intervening_tokens = []
                    for k in range(current_atom_index,0,-1):
                        if tokens[k] == "(":
                            break
                        intervening_tokens.append(tokens[k])

                    # Check for double bond (=)
                    if '=' in intervening_tokens:
                        adjmat[i, j] = adjmat[j, i] = 2
                    # Check for triple bond (#)
                    elif '#' in intervening_tokens:
                        adjmat[i, j] = adjmat[j, i] = 3
                    # Update the adjacency matrix for a single bond                
                    else:
                        adjmat[i, j] = adjmat[j, i] = 1                        
                        
                    # break out of the search for branch closures
                    break

    # Iterate through the branch ending points and add bonds. 
    for i in branch_close_atom_indices:

        # Current and next atom indices in the tokens list
        current_atom_index = atom_indices[i]
        current_branch_level = branch_levels[i]
        
        # Look backwards until finding an atom at the same branch level
        for j in range(i-1,-1,-1):

            if branch_levels[j] == current_branch_level:

                # Extract intervening tokens between the branch end token and the current atom
                intervening_tokens = []
                for k in range(current_atom_index,0,-1):
                    if tokens[k] == ")":
                        break
                    intervening_tokens.append(tokens[k])

                # Check for double bond (=)
                if '=' in intervening_tokens:
                    adjmat[i, j] = adjmat[j, i] = 2
                # Check for triple bond (#)
                elif '#' in intervening_tokens:
                    adjmat[i, j] = adjmat[j, i] = 3
                # Update the adjacency matrix for a single bond                
                else:
                    adjmat[i, j] = adjmat[j, i] = 1                        

                # break out of the search for branch closures
                break
    
    # Handle aromatics (BMS: aromatic symbols should be discontinued, they are fragile and poorly defined)
    aromatics = {"b","c","n","o","p","s"}
    if any([ _ in smiles2adjmat.aromatics for _ in tokens ]):

        # Loop over all rings
        for r in return_rings(adjmat_to_adjlist(adjmat),max_size=10,remove_fused=True):
    
            # If all the atoms in the ring have aromatic labels then the alternating double bonds are placed
            if all([ atom_info[_][0] in smiles2adjmat.aromatics for _ in r ]):

                # Place all double-bonds less the final connection
                for ind in range(1,len(r)):
                    if (ind-1)%2==0:
                        adjmat[r[ind],r[ind-1]]=2
                        adjmat[r[ind-1],r[ind]]=2

    # Find atoms with explicit hydrogens attached to them via the adjacency matrix and set their explicit H count to 0
    # This is to avoid hydrogen inference by add_hydrogens(), assuming that the user knowingly supplied the explicit Hs                    
    explicit_h = { count_j for count_i,i in enumerate(atom_info) if i[0].lower() == "h" for count_j,j in enumerate(adjmat[count_i]) if j != 0 }
    for i in range(len(atom_info)):
        if i in explicit_h and not atom_info[i][3]:
            atom_info[i][2] = 0
                        
    # Add hydrogens
    adjmat,atom_info = add_hydrogens(adjmat,atom_info)
    return adjmat, atom_info

def add_hydrogens(adjmat, atom_info):
    """
    This is a helper function for the smiles2adjmat() function that adds hydrogens to atoms based on either
    the explicit number of hydrogens designation or the using an inference algorithm based on the formal charge
    and number of bonds. If an explicit number of hydrogens is supplied it will overrule the hydrogen inference
    routine based on formal charge. Formal charge inference is based on the neutral full octet protonation state
    with protons/hydrides removed or added to meet the formal charge requirements.
 
    Parameters
    ----------
    adjmat: array
            This is numpy array holding the graph defined by the smiles string.

    atom_info: list of tuples
            This list is indexed to the adjacency matrix and contains a tuple for each atom. Each tuple has
            the element token, formal charge, explicit hydrogens, and isotope information as its respective
            elements. 

    Returns
    -------

    adjmat: array
            This is numpy array holding the graph defined by the smiles string. This array is expanded relative
            to the inputted array to accomodate the additional hydrogens. 

    atom_info: list of tuples
            This list is indexed to the adjacency matrix and contains a tuple for each atom. Each tuple has
            the element token, formal charge, explicit hydrogens, and isotope information as its respective
            elements. This list is expanded relative to the inputted list to reflect the additional hydrogens.

    """
            
    # Number of hydrogens to add for each atom
    hydrogens_to_add = []

    # Loop over the atoms and determine how many hydrogens to add
    for atom, info in enumerate(atom_info):

        # Determine the element of the current atom
        element = info[0].lower()
        formal_charge = info[1]
        explicit_hydrogens = info[2] # hydrogens designated via [] annotation
        
        # Calculate the number of valence electrons
        valence_electrons = el_valence.get(element, None)

        # If the atomic valence is undefined then assume no hydrogens need to be added.
        if valence_electrons is None:
            print(f"Warning: Element '{element}' is not recognized or has an undefined valence.")
            hydrogens_to_add.append(0)
            continue
        # If the atom is a metal then it isn't hydrogenated
        elif element in el_metals:
            hydrogens_to_add.append(0)
            continue

        # Calculate the number of bonds (each bond contributes 1 electron)
        bonds = sum(adjmat[atom])

        # If the number of explicit hydrogens is given, use that
        if explicit_hydrogens is not None:
            needed_hydrogens = explicit_hydrogens
            
        # Otherwise, calculate the number of hydrogens to be added based on the formal charge
        else:
            
            # Determine the desired number of electrons for a full octet (8 for most elements)
            # Special cases like hydrogen (which wants 2) can be handled here
            desired_electrons = 8 if element not in ['h', 'he'] else 2

            # Calculate the current electrons: valence electrons + bonds - charge
            current_electrons = int(valence_electrons + bonds)

            # Add hydrogen radicals to reach the desired number of electrons
            needed_hydrogens = max(0, desired_electrons - current_electrons)
            
            # Cation case
            if formal_charge > 0:
                e = desired_electrons - 2*needed_hydrogens - 2*bonds # number of unbound electrons
                
                # Case where the formal charge can be satisfied entirely by adding protons to lone-pairs
                if (formal_charge - int(e/2)) <= 0:
                    needed_hydrogens += formal_charge
                
                # Case where the formal charge requires the loss of hydride
                else:
                    needed_hydrogens -= formal_charge
                    
            # Anion case
            elif formal_charge < 0:
                e = desired_electrons - 2*needed_hydrogens - 2*bonds # number of unbound electrons

                # Case where the formal charge can be satisfied entirely by loss of protons (leaving behind lone pairs)
                if (needed_hydrogens + formal_charge)>=0:
                    needed_hydrogens += formal_charge
                    
                # Case where the formal charge requires the addition of hydride
                else:
                    needed_hydrogens -= formal_charge
                
        # Check that not all electrons have been removed
        if needed_hydrogens < 0:
            print(f"Warning: add_hydrogens() was unable to satisfy formal charge specification with hydrogens.")
        # Check that the octet has not been violated
        if (2*bonds + 2*needed_hydrogens) > 8 and not el_expand_octet[element]:
            raise OctetError(atom)
    
        # Update list of hydrogens to be added
        hydrogens_to_add.append(needed_hydrogens)

    # Create a new adjacency matrix including hydrogens
    total_atoms = len(hydrogens_to_add) + int(sum(hydrogens_to_add))
    new_adjmat = np.zeros((total_atoms, total_atoms))

    # Fill in the original adjmat values
    new_adjmat[:len(adjmat), :len(adjmat)] = adjmat
    
    # Add hydrogens to the adjacency matrix
    current_index = len(adjmat)
    for i, num_hydrogens in enumerate(hydrogens_to_add):
        for _ in range(num_hydrogens):
            new_adjmat[i, current_index] = new_adjmat[current_index, i] = 1
            current_index += 1

    # Add hydrogens to atom_info in the return statement
    return new_adjmat,atom_info+[('h',0,None,None) for _ in range(int(sum(hydrogens_to_add)))]

# Define the custom exception
class OctetError(ValueError):
    """Exception raised when the number of electrons exceeds the allowed limit."""
    def __init__(self, atom_indices, message="Atom indices {} has an octet violation."):
        self.message = message.format(atom_indices)
        super().__init__(self.message)
