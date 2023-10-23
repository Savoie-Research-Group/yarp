"""
This module holds useful properties dictionaries used by other objects in the yarp package
"""

# element label to atomic number
el_to_an = { "h": 1,  "he": 2,\
             "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
             "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
             "k":19, "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
             "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
             "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}
# add values for title case
for _ in list(el_to_an.keys()):
    el_to_an[_.title()] = el_to_an[_]

# atomic number to element (lower-case)
an_to_el = { el_to_an[i]:i.lower() for i in el_to_an.keys() }

# Used for determining number of valence electrons provided by each atom to a neutral molecule when calculating Lewis structures
el_valence = {  'h':1, 'he':2,\
                'li':1, 'be':2,                                                                                                                'b':3,  'c':4,  'n':5,  'o':6,  'f':7, 'ne':8,\
                'na':1, 'mg':2,                                                                                                               'al':3, 'si':4,  'p':5,  's':6, 'cl':7, 'ar':8,\
                'k' :1, 'ca':2, 'sc':None, 'ti':None,  'v':None, 'cr':None, 'mn':None, 'fe':3, 'co':None, 'ni':None, 'cu':None, 'zn':None, 'ga':3, 'ge':4, 'as':5, 'se':6, 'br':7, 'kr':8,\
                'rb':1, 'sr':2,  'y':None, 'zr':None, 'nb':None, 'mo':None, 'tc':None, 'ru':None, 'rh':None, 'pd':None, 'ag':None, 'cd':None, 'in':3, 'sn':4, 'sb':5, 'te':6,  'i':7, 'xe':8,\
                'cs':1, 'ba':2, 'la':None, 'hf':None, 'ta':None,  'w':None, 're':None, 'os':None, 'ir':None, 'pt':None, 'au':None, 'hg':None, 'tl':3, 'pb':4, 'bi':5, 'po':6, 'at':7, 'rn':8  }        
# add values for title case
for _ in list(el_valence.keys()):
    el_valence[_.title()] = el_valence[_]

# Used for determining electron deficiency when calculating lewis structures
el_n_deficient = {  'h':2, 'he':2,\
                    'li':0, 'be':0,                                                                                                                'b':8,  'c':8,  'n':8,  'o':8,  'f':8, 'ne':8,\
                    'na':0, 'mg':0,                                                                                                               'al':8, 'si':8,  'p':8,  's':8, 'cl':8, 'ar':8,\
                    'k' :0, 'ca':0, 'sc':None, 'ti':None,  'v':None, 'cr':None, 'mn':None, 'fe':0, 'co':None, 'ni':None, 'cu':None, 'zn':None, 'ga':8, 'ge':8, 'as':8, 'se':8, 'br':8, 'kr':8,\
                    'rb':0, 'sr':0,  'y':None, 'zr':None, 'nb':None, 'mo':None, 'tc':None, 'ru':None, 'rh':None, 'pd':None, 'ag':None, 'cd':None, 'in':8, 'sn':8, 'sb':8, 'te':8,  'i':8, 'xe':8,\
                    'cs':0, 'ba':0, 'la':None, 'hf':None, 'ta':None,  'w':None, 're':None, 'os':None, 'ir':None, 'pt':None, 'au':None, 'hg':None, 'tl':8, 'pb':8, 'bi':8, 'po':8, 'at':8, 'rn':8  }        
# add values for title case
for _ in list(el_n_deficient.keys()):
    el_n_deficient[_.title()] = el_n_deficient[_]

# Used to determine is expanded octets are allowed when calculating Lewis structures
el_expand_octet = { 'h':False, 'he':False,\
                    'li':False, 'be':False,                                                                                                               'b':False,  'c':False, 'n':False, 'o':False, 'f':False,'ne':False,\
                    'na':False, 'mg':False,                                                                                                               'al':True, 'si':True,  'p':True,  's':True, 'cl':True, 'ar':True,\
                    'k' :False, 'ca':False, 'sc':None, 'ti':None,  'v':None, 'cr':None, 'mn':None, 'fe':True, 'co':None, 'ni':None, 'cu':None, 'zn':None, 'ga':True, 'ge':True, 'as':True, 'se':True, 'br':True, 'kr':True,\
                    'rb':False, 'sr':False,  'y':None, 'zr':None, 'nb':None, 'mo':None, 'tc':None, 'ru':None, 'rh':None, 'pd':None, 'ag':None, 'cd':None, 'in':True, 'sn':True, 'sb':True, 'te':True,  'i':True, 'xe':True,\
                    'cs':False, 'ba':False, 'la':None, 'hf':None, 'ta':None,  'w':None, 're':None, 'os':None, 'ir':None, 'pt':None, 'au':None, 'hg':None, 'tl':True, 'pb':True, 'bi':True, 'po':True, 'at':True, 'rn':True  }
# add values for title case
for _ in list(el_expand_octet.keys()):
    el_expand_octet[_.title()] = el_expand_octet[_]

# Electronegativity (Allen scale)
el_en = { "h" :2.3,  "he":4.16,\
          "li":0.91, "be":1.58,                                                                                                               "b" :2.05, "c" :2.54, "n" :3.07, "o" :3.61, "f" :4.19, "ne":4.79,\
          "na":0.87, "mg":1.29,                                                                                                               "al":1.61, "si":1.91, "p" :2.25, "s" :2.59, "cl":2.87, "ar":3.24,\
          "k" :0.73, "ca":1.03, "sc":1.19, "ti":1.38, "v": 1.53, "cr":1.65, "mn":1.75, "fe":1.80, "co":1.84, "ni":1.88, "cu":1.85, "zn":1.59, "ga":1.76, "ge":1.99, "as":2.21, "se":2.42, "br":2.69, "kr":2.97,\
          "rb":0.71, "sr":0.96, "y" :1.12, "zr":1.32, "nb":1.41, "mo":1.47, "tc":1.51, "ru":1.54, "rh":1.56, "pd":1.58, "ag":1.87, "cd":1.52, "in":1.66, "sn":1.82, "sb":1.98, "te":2.16, "i" :2.36, "xe":2.58,\
          "cs":0.66, "ba":0.88, "la":1.09, "hf":1.16, "ta":1.34, "w" :1.47, "re":1.60, "os":1.65, "ir":1.68, "pt":1.72, "au":1.92, "hg":1.76, "tl":1.79, "pb":1.85, "bi":2.01, "po":2.19, "at":2.39, "rn":2.60} 
# add values for title case
for _ in list(el_en.keys()):
    el_en[_.title()] = el_en[_]

# Polarizability ordering (for determining lewis structure)
el_pol ={ "h" :4.5,  "he":1.38,\
          "li":164.0, "be":377,                                                                                                               "b" :20.5, "c" :11.3, "n" :7.4, "o" :5.3,  "f" :3.74, "ne":2.66,\
          "na":163.0, "mg":71.2,                                                                                                              "al":57.8, "si":37.3, "p" :25.0,"s" :19.4, "cl":14.6, "ar":11.1,\
          "k" :290.0, "ca":161.0, "sc":97.0, "ti":100.0, "v": 87.0, "cr":83.0, "mn":68.0, "fe":62.0, "co":55, "ni":49, "cu":47.0, "zn":38.7,  "ga":50.0, "ge":40.0, "as":30.0,"se":29.0, "br":21.0, "kr":16.8,\
          "rb":320.0, "sr":197.0, "y" :162,  "zr":112.0, "nb":98.0, "mo":87.0, "tc":79.0, "ru":72.0, "rh":66, "pd":26.1, "ag":55, "cd":46.0,  "in":65.0, "sn":53.0, "sb":43.0,"te":28.0, "i" :32.9, "xe":27.3,}
# add values for title case
for _ in list(el_pol.keys()):
    el_pol[_.title()] = el_pol[_]


# Average atomic masses
el_mass = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                     'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                     'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                     'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                     'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                     'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                     'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                     'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}
# add values for lower case
for _ in list(el_mass.keys()):
    el_mass[_.lower()] = el_mass[_]


# Atomic radii based on UFF (Rappe et al. JACS 1992) but with some tweaking based on experience. These were developed for parsing
# bonds based on atomic separations.    
# These radii neglect the bond-order and electronegativity corrections in the original paper. Where several values exist
# the largest was used. All units are in angstroms.
el_radii = {  'H':0.39, 'He':0.849,\
              'Li':1.336, 'Be':1.074,                                                                                                                          'B':0.838,  'C':0.757,  'N':0.700,  'O':0.658,  'F':0.668, 'Ne':0.920,\
              'Na':1.539, 'Mg':1.421,                                                                                                                         'Al':1.15,  'Si':1.050,  'P':1.117,  'S':1.064, 'Cl':1.044, 'Ar':1.032,\
              'K' :1.953, 'Ca':1.761, 'Sc':1.513, 'Ti':1.412,  'V':1.402, 'Cr':1.345, 'Mn':1.382, 'Fe':1.335, 'Co':1.241, 'Ni':1.164, 'Cu':1.302, 'Zn':1.193, 'Ga':1.260, 'Ge':1.197, 'As':1.211, 'Se':1.190, 'Br':1.192, 'Kr':1.147,\
              'Rb':2.260, 'Sr':2.052,  'Y':1.698, 'Zr':1.564, 'Nb':1.400, 'Mo':1.484, 'Tc':1.322, 'Ru':1.478, 'Rh':1.332, 'Pd':1.338, 'Ag':1.386, 'Cd':1.403, 'In':1.459, 'Sn':1.398, 'Sb':1.407, 'Te':1.386,  'I':1.382, 'Xe':1.267,\
              'Cs':2.570, 'Ba':2.277, 'La':1.943, 'Hf':1.611, 'Ta':1.511,  'W':1.526, 'Re':1.372, 'Os':1.372, 'Ir':1.371, 'Pt':1.364, 'Au':1.262, 'Hg':1.340, 'Tl':1.518, 'Pb':1.459, 'Bi':1.512, 'Po':1.500, 'At':1.545, 'Rn':1.42,\
              'default' : 0.7 }
# add values for lower case
for _ in list(el_radii.keys()):
    el_radii[_.lower()] = el_radii[_]

# This dictionary is used to flagging problematic adjacency matrices by the table_generator function.
el_max_bonds = {  'H':2,    'He':1,\
                  'Li':None, 'Be':None,                                                                                                                'B':4,     'C':4,     'N':4,     'O':2,     'F':1,    'Ne':1,\
                  'Na':None, 'Mg':None,                                                                                                               'Al':4,    'Si':4,  'P':None,  'S':None, 'Cl':1,    'Ar':1,\
                  'K' :None, 'Ca':None, 'Sc':15, 'Ti':14,  'V':13, 'Cr':12, 'Mn':11, 'Fe':10, 'Co':9, 'Ni':8, 'Cu':None, 'Zn':None, 'Ga':3,    'Ge':None, 'As':None, 'Se':None, 'Br':1,    'Kr':None,\
                  'Rb':None, 'Sr':None,  'Y':15, 'Zr':14, 'Nb':13, 'Mo':12, 'Tc':11, 'Ru':10, 'Rh':9, 'Pd':8, 'Ag':None, 'Cd':None, 'In':None, 'Sn':None, 'Sb':None, 'Te':None,  'I':1,    'Xe':None,\
                  'Cs':None, 'Ba':None, 'La':15, 'Hf':14, 'Ta':13,  'W':12, 'Re':11, 'Os':10, 'Ir':9, 'Pt':8, 'Au':None, 'Hg':None, 'Tl':None, 'Pb':None, 'Bi':None, 'Po':None, 'At':None, 'Rn':None  }
# add values for lower case
for _ in list(el_max_bonds.keys()):
    el_max_bonds[_.lower()] = el_max_bonds[_]

# This dictionary is used to flagging problematic adjacency matrices by the table_generator function.
el_max_bonds = {  'H':2,    'He':1,\
                  'Li':None, 'Be':None,                                                                                                                'B':4,     'C':4,     'N':4,     'O':2,     'F':1,    'Ne':1,\
                  'Na':None, 'Mg':None,                                                                                                               'Al':4,    'Si':4,  'P':None,  'S':None, 'Cl':1,    'Ar':1,\
                  'K' :None, 'Ca':None, 'Sc':15, 'Ti':14,  'V':13, 'Cr':12, 'Mn':11, 'Fe':10, 'Co':9, 'Ni':8, 'Cu':None, 'Zn':None, 'Ga':3,    'Ge':None, 'As':None, 'Se':None, 'Br':1,    'Kr':None,\
                  'Rb':None, 'Sr':None,  'Y':15, 'Zr':14, 'Nb':13, 'Mo':12, 'Tc':11, 'Ru':10, 'Rh':9, 'Pd':8, 'Ag':None, 'Cd':None, 'In':None, 'Sn':None, 'Sb':None, 'Te':None,  'I':1,    'Xe':None,\
                  'Cs':None, 'Ba':None, 'La':15, 'Hf':14, 'Ta':13,  'W':12, 'Re':11, 'Os':10, 'Ir':9, 'Pt':8, 'Au':None, 'Hg':None, 'Tl':None, 'Pb':None, 'Bi':None, 'Po':None, 'At':None, 'Rn':None  }
# add values for lower case
for _ in list(el_max_bonds.keys()):
    el_max_bonds[_.lower()] = el_max_bonds[_]


# This dictionary is used to flagging problematic adjacency matrices by the table_generator function.
el_max_valence = {  'H':2,    'He':2,\
                   'Li':2, 'Be':100,                                                                                                      'B':4,    'C':4,    'N':4,    'O':4,    'F':4,   'Ne':4,\
                   'Na':2, 'Mg':100,                                                                                                     'Al':100, 'Si':100,  'P':6,  'S':4, 'Cl':100, 'Ar':100,\
                   'K' :2, 'Ca':100, 'Sc':100, 'Ti':100,  'V':100, 'Cr':100, 'Mn':100, 'Fe':6, 'Co':100, 'Ni':100, 'Cu':100, 'Zn':100, 'Ga':100, 'Ge':100, 'As':100, 'Se':4, 'Br':100, 'Kr':100,\
                   'Rb':2, 'Sr':100,  'Y':100, 'Zr':100, 'Nb':100, 'Mo':100, 'Tc':100, 'Ru':100, 'Rh':100, 'Pd':100, 'Ag':100, 'Cd':100, 'In':100, 'Sn':100, 'Sb':100, 'Te':100,  'I':100, 'Xe':100,\
                   'Cs':2, 'Ba':100, 'La':100, 'Hf':100, 'Ta':100,  'W':100, 'Re':100, 'Os':100, 'Ir':100, 'Pt':100, 'Au':100, 'Hg':100, 'Tl':100, 'Pb':100, 'Bi':100, 'Po':100, 'At':100, 'Rn':100  }
# add values for lower case
for _ in list(el_max_valence.keys()):
    el_max_valence[_.lower()] = el_max_valence[_]

    
