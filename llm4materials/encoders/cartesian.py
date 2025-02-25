from ase import Atoms

class Cartesian:
    def __init__(self):
        pass
        
    def encode(self, atoms):
        return struct2cartesian(atoms)
    
    def decode(self, string):
        return cartesian2struct(string)
    
    @property
    def prompt_header(self):
        return (
        "Below is a description of a bulk material. "
        "Generate a description of the lengths and angles of the lattice vectors "
        "and then the element type and coordinates for each atom within the lattice:\n"
    )

    @property
    def conditional_prompt_header(self):
        return (
        "Below is a description of a bulk material. "
        "The formation energy per atom is <eform> eV/atom, "
        "the bandgap is <bandgap> eV. "
        "Generate a description of the lengths and angles of the lattice vectors "
        "and then the element type and coordinates for each atom within the lattice:\n"
    )

def struct2cartesian(
    atoms,
    decimals=2,
    fractional_coors=True
):
    """
    Given the atomic symbols, positions and cell of a structure,
    return a string representation of the structure (CIF).

    Args:
        fractional_coors (bool): Whether to use fractional coordinates or not.
    """
    atomic_symbols = atoms.get_chemical_symbols()
    lattice_params = atoms.cell.cellpar()
    lengths = lattice_params[:3]
    angles = lattice_params[3:]
    coors = atoms.get_scaled_positions() if fractional_coors else atoms.get_positions()

    # Create the CIF string
    cif_str = \
        " ".join(["{0:.1f}".format(x) for x in lengths]) + "\n" + \
        " ".join([str(int(x)) for x in angles]) + "\n" + \
        "\n".join([
            str(t) + "\n" + " ".join([
                "{0:.2f}".format(x) for x in c
            ]) for t,c in zip(atomic_symbols, coors)
        ])
    
    return cif_str

def cartesian2struct(cif_str: str):
    lines = cif_str.split("\n")

    # cell
    cell = []
    for l in lines[:2]:
        cell += [float(x) for x in l.split(" ")]
    
    # atomic symbols and positions
    atomic_symbols = []
    positions = []

    for l in lines[2:]:
        if not l:
            continue
        
        if l.isalpha():
            atomic_symbols.append(l)
        else:
            positions.append([float(x) for x in l.split(" ")])

    # construct atoms object
    atoms = Atoms(
        symbols=atomic_symbols,
        scaled_positions=positions,
        cell=cell,
        pbc=[True, True, True]
    )
    
    return atoms