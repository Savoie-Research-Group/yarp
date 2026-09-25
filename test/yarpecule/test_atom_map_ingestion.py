from pathlib import Path

import numpy as np
import pytest

from yarp.yarpecule.input_parsers import (
    load_reaction_from_xyz_file,
    load_reactions_from_smiles_file,
    reaction_xyz_parse,
    xyz_parse,
)
from yarp.yarpecule.yarpecule import yarpecule


ROOT = Path(__file__).resolve().parents[2]
XYZ_SPECIES = ROOT / "test" / "molecules" / "ethene.xyz"
XYZ_REACTION = ROOT / "test" / "molecules" / "batch_xyz_rxn" / "reaction1.xyz"


def atom_maps(molecule):
    return [
        molecule._atom_info[i]["atom_map"]
        for i in range(len(molecule.elements))
    ]


def geometry_by_map(molecule):
    return {
        molecule._atom_info[i]["atom_map"]: molecule.geo[i]
        for i in range(len(molecule.elements))
    }


def test_tuple_input_requires_atom_info_container():
    adjacency = np.zeros((1, 1), dtype=int)
    geometry = np.zeros((1, 3))
    core = (adjacency, geometry, ["h"], 0)

    with pytest.raises(TypeError):
        yarpecule(core, canon=False, strict=True)

    molecule = yarpecule((*core, {}), canon=False)
    assert atom_maps(molecule) == [0]


def write_xyz(path, elements, geo, comment=""):
    lines = [str(len(elements)), comment]
    lines.extend(
        f"{element} {x:.12f} {y:.12f} {z:.12f}"
        for element, (x, y, z) in zip(elements, geo)
    )
    path.write_text("\n".join(lines) + "\n")


def write_xyz_reaction(
    path,
    reactant_elements,
    reactant_geo,
    product_elements,
    product_geo,
    charge,
):
    lines = []
    for elements, geo in (
        (reactant_elements, reactant_geo),
        (product_elements, product_geo),
    ):
        lines.extend((str(len(elements)), f"q {charge}"))
        lines.extend(
            f"{element} {x:.12f} {y:.12f} {z:.12f}"
            for element, (x, y, z) in zip(elements, geo)
        )
    path.write_text("\n".join(lines) + "\n")


def test_xyz_species_maps_follow_file_order():
    input_elements, input_geo = xyz_parse(XYZ_SPECIES)
    molecule = yarpecule(str(XYZ_SPECIES))

    maps = atom_maps(molecule)
    assert sorted(maps) == list(range(len(input_elements)))
    for local_index, atom_map in enumerate(maps):
        assert molecule.elements[local_index] == input_elements[atom_map].lower()
        assert np.allclose(molecule.geo[local_index], input_geo[atom_map])


def test_xyz_species_scrambled_rows_change_map_correspondence(tmp_path):
    input_elements, input_geo = xyz_parse(XYZ_SPECIES)
    order = [1, 0, 3, 2, 5, 4]
    scrambled_path = tmp_path / "scrambled_species.xyz"
    write_xyz(
        scrambled_path,
        [input_elements[i] for i in order],
        input_geo[order],
    )

    original = yarpecule(str(XYZ_SPECIES))
    scrambled = yarpecule(str(scrambled_path))
    original_by_map = geometry_by_map(original)
    scrambled_by_map = geometry_by_map(scrambled)

    assert sorted(atom_maps(scrambled)) == list(range(len(order)))
    assert np.allclose(scrambled_by_map[0], input_geo[order[0]])
    assert not np.allclose(scrambled_by_map[0], original_by_map[0])


def test_xyz_reaction_endpoints_share_file_order_maps():
    rxn = load_reaction_from_xyz_file(XYZ_REACTION)
    expected_maps = list(range(len(rxn.reactant.graph.elements)))

    assert atom_maps(rxn.reactant.graph) == expected_maps
    assert atom_maps(rxn.product.graph) == expected_maps
    assert rxn.hash is not None


def test_xyz_reaction_scrambled_rows_change_map_correspondence(tmp_path):
    (
        reactant_elements,
        reactant_geo,
        reactant_q,
        _,
        product_elements,
        product_geo,
        product_q,
        _,
    ) = reaction_xyz_parse(XYZ_REACTION)
    assert reactant_q == product_q

    order = list(reversed(range(len(reactant_elements))))
    scrambled_path = tmp_path / "scrambled_reaction.xyz"
    write_xyz_reaction(
        scrambled_path,
        [reactant_elements[i] for i in order],
        reactant_geo[order],
        [product_elements[i] for i in order],
        product_geo[order],
        reactant_q,
    )

    original = load_reaction_from_xyz_file(XYZ_REACTION)
    scrambled = load_reaction_from_xyz_file(scrambled_path)
    original_by_map = geometry_by_map(original.reactant.graph)
    scrambled_by_map = geometry_by_map(scrambled.reactant.graph)
    expected_maps = list(range(len(order)))

    assert atom_maps(scrambled.reactant.graph) == expected_maps
    assert atom_maps(scrambled.product.graph) == expected_maps
    assert np.allclose(scrambled_by_map[0], reactant_geo[order[0]])
    assert not np.allclose(scrambled_by_map[0], original_by_map[0])
    assert scrambled.hash == original.hash


def test_smiles_species_preserves_input_atom_maps():
    molecule = yarpecule("[O:41]([H:7])[C:99]([H:8])([H:9])[H:10]")
    element_by_map = {
        molecule._atom_info[i]["atom_map"]: molecule.elements[i]
        for i in range(len(molecule.elements))
    }

    assert element_by_map == {
        7: "h",
        8: "h",
        9: "h",
        10: "h",
        41: "o",
        99: "c",
    }


@pytest.mark.parametrize("canon", [False, True])
def test_partially_mapped_smiles_uses_one_atom_map_field(canon):
    molecule = yarpecule("[C:41]([H])([H])([H])[O:99][H]", canon=canon)
    maps = atom_maps(molecule)
    element_by_map = dict(zip(maps, molecule.elements))

    assert element_by_map[41] == "c"
    assert element_by_map[99] == "o"
    assert len(maps) == len(set(maps))
    assert all("input_atom_map" not in info for info in molecule._atom_info.values())
    assert all("atom_index" not in info for info in molecule._atom_info.values())


def test_smiles_species_scrambled_labels_change_output_maps():
    original = yarpecule("[O:41]([H:7])[C:99]([H:8])([H:9])[H:10]")
    scrambled = yarpecule("[H:310][O:341][C:399]([H:307])([H:308])[H:309]")
    scrambled_element_by_map = {
        scrambled._atom_info[i]["atom_map"]: scrambled.elements[i]
        for i in range(len(scrambled.elements))
    }

    assert set(scrambled_element_by_map) != set(atom_maps(original))
    assert scrambled_element_by_map == {
        307: "h",
        308: "h",
        309: "h",
        310: "h",
        341: "o",
        399: "c",
    }
    assert scrambled.hash == original.hash


def test_smiles_reaction_endpoints_preserve_matching_maps(tmp_path):
    source = tmp_path / "reaction.smi"
    source.write_text(
        "[C:11]([H:21])([H:22])([H:23])[O:12][H:24]>>"
        "[H:24][O:12][C:11]([H:21])([H:22])[H:23]\n"
    )
    rxn = next(iter(load_reactions_from_smiles_file(source).values()))

    assert set(atom_maps(rxn.reactant.graph)) == {11, 12, 21, 22, 23, 24}
    assert set(atom_maps(rxn.product.graph)) == {11, 12, 21, 22, 23, 24}
    assert rxn.hash is not None


def test_smiles_reaction_scrambled_labels_change_endpoint_maps(tmp_path):
    original_path = tmp_path / "original_reaction.smi"
    original_path.write_text(
        "[C:11]([H:21])([H:22])([H:23])[O:12][H:24]>>"
        "[H:24][O:12][C:11]([H:21])([H:22])[H:23]\n"
    )
    scrambled_path = tmp_path / "scrambled_reaction.smi"
    scrambled_path.write_text(
        "[C:211]([H:321])([H:322])([H:323])[O:212][H:324]>>"
        "[H:324][O:212][C:211]([H:321])([H:322])[H:323]\n"
    )

    original = next(iter(load_reactions_from_smiles_file(original_path).values()))
    scrambled = next(iter(load_reactions_from_smiles_file(scrambled_path).values()))
    scrambled_reactant_maps = set(atom_maps(scrambled.reactant.graph))
    scrambled_product_maps = set(atom_maps(scrambled.product.graph))

    assert scrambled_reactant_maps != set(atom_maps(original.reactant.graph))
    assert scrambled_reactant_maps == scrambled_product_maps
    assert scrambled_reactant_maps == {211, 212, 321, 322, 323, 324}
    assert scrambled.hash == original.hash
