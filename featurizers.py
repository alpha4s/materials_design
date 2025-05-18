# featurizers.py
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

comp_featurizer = ElementProperty.from_preset("magpie")

def featurize_composition(formula: str):
    """MagPie composition vector for a formula string."""
    comp = Composition(formula)
    return comp_featurizer.featurize(comp)
