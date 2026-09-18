"""Fixed, versioned AME2020 + NIST/CODATA2022 + PDG2026 mass catalog.

Masses below are represented by mc^2 in eV. Displaying the same number as a
mass requires eV/c^2. Ion masses include cumulative ASD ionization energies;
the element-level ASD data do not resolve isotope shifts or nuclear isomers.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path


@lru_cache(maxsize=1)
def load_mass_catalog():
    catalog_path = Path(__file__).with_name("mass_catalog.json")
    try:
        return json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取随包提供的权威质量数据：{exc}") from exc


@dataclass(frozen=True)
class MassRecord:
    energy_ev: float
    reference: str
    note: str = ""


def constant_mass(name):
    return load_mass_catalog()["constants"][name]["mass_mev"] * 1e6


def special_mass(key):
    constants = {"e": "electron", "mu": "muon", "p": "proton", "n": "neutron"}
    if key in constants:
        return MassRecord(constant_mass(constants[key]), "NIST CODATA 2022")
    pdg_id = {"tau": "15", "pi": "211", "pi0": "111", "K": "321", "K0": "311"}[key]
    return MassRecord(load_mass_catalog()["pdg"][pdg_id]["mass_mev"] * 1e6, "PDG 2026")


@lru_cache(maxsize=4096)
def ion_mass(a, z, q):
    if z is None:
        raise ValueError("精确离子质量需要质子数 Z；请填写 Z 或使用粒子搜索。")
    direct = {(1, 1, 1): "proton", (2, 1, 1): "deuteron", (3, 1, 1): "triton", (3, 2, 2): "helion", (4, 2, 2): "alpha particle"}.get((a, z, q))
    if direct:
        return MassRecord(constant_mass(direct), "NIST CODATA 2022")
    catalog = load_mass_catalog()
    atom = catalog["isotopes"].get(f"{a},{z}")
    if atom is None:
        raise ValueError("AME2020 未收录此同位素的基态质量，请检查 A、Z。")
    energy = atom["mass_u"] * constant_mass("atomic mass constant") - q * constant_mass("electron")
    note = "AME2020 标记为估算质量。" if atom["estimated"] else ""
    if q < 0:
        if (a, z, q) != (1, 1, -1):
            raise ValueError("当前固定数据仅支持 H− 负离子；其他负离子缺少电子亲和能。")
        energy -= catalog["metadata"]["hydrogen_affinity"]["energy_ev"]
        return MassRecord(energy, "AME2020 + CODATA2022 + NIST WebBook", note)
    stages = [catalog["ionization"].get(f"{z},{j}") for j in range(q)]
    if any(r is None or r["energy_ev"] is None for r in stages):
        raise ValueError("NIST ASD 缺少此电荷态所需的电离能，无法可靠计算离子质量。")
    energy += math.fsum(r["energy_ev"] for r in stages)
    if stages:
        note += "已计入电离能；未计同位素位移。"
    return MassRecord(energy, "AME2020 + CODATA2022 + NIST ASD", note)
