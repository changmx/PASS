"""Offline element lookup and PASS-compatible reference-particle identities.

Element names/symbols follow the IUPAC periodic table:
https://iupac.org/what-we-do/periodic-table-of-elements/
The name catalog does not certify isotope existence. Fixed evaluated masses
come from AME2020, NIST/CODATA2022 and PDG2026; see particle_masses.py.
"""
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
import unicodedata

from PASS.tool.particle_masses import constant_mass, ion_mass, special_mass

# Atomic-number order. The mass convention comes from PASS, not atomic weights.
_ELEMENT_DATA = """
H hydrogen 氢
He helium 氦
Li lithium 锂
Be beryllium 铍
B boron 硼
C carbon 碳
N nitrogen 氮
O oxygen 氧
F fluorine 氟
Ne neon 氖
Na sodium 钠
Mg magnesium 镁
Al aluminium 铝
Si silicon 硅
P phosphorus 磷
S sulfur 硫
Cl chlorine 氯
Ar argon 氩
K potassium 钾
Ca calcium 钙
Sc scandium 钪
Ti titanium 钛
V vanadium 钒
Cr chromium 铬
Mn manganese 锰
Fe iron 铁
Co cobalt 钴
Ni nickel 镍
Cu copper 铜
Zn zinc 锌
Ga gallium 镓
Ge germanium 锗
As arsenic 砷
Se selenium 硒
Br bromine 溴
Kr krypton 氪
Rb rubidium 铷
Sr strontium 锶
Y yttrium 钇
Zr zirconium 锆
Nb niobium 铌
Mo molybdenum 钼
Tc technetium 锝
Ru ruthenium 钌
Rh rhodium 铑
Pd palladium 钯
Ag silver 银
Cd cadmium 镉
In indium 铟
Sn tin 锡
Sb antimony 锑
Te tellurium 碲
I iodine 碘
Xe xenon 氙
Cs caesium 铯
Ba barium 钡
La lanthanum 镧
Ce cerium 铈
Pr praseodymium 镨
Nd neodymium 钕
Pm promethium 钷
Sm samarium 钐
Eu europium 铕
Gd gadolinium 钆
Tb terbium 铽
Dy dysprosium 镝
Ho holmium 钬
Er erbium 铒
Tm thulium 铥
Yb ytterbium 镱
Lu lutetium 镥
Hf hafnium 铪
Ta tantalum 钽
W tungsten 钨
Re rhenium 铼
Os osmium 锇
Ir iridium 铱
Pt platinum 铂
Au gold 金
Hg mercury 汞
Tl thallium 铊
Pb lead 铅
Bi bismuth 铋
Po polonium 钋
At astatine 砹
Rn radon 氡
Fr francium 钫
Ra radium 镭
Ac actinium 锕
Th thorium 钍
Pa protactinium 镤
U uranium 铀
Np neptunium 镎
Pu plutonium 钚
Am americium 镅
Cm curium 锔
Bk berkelium 锫
Cf californium 锎
Es einsteinium 锿
Fm fermium 镄
Md mendelevium 钔
No nobelium 锘
Lr lawrencium 铹
Rf rutherfordium 𬬻
Db dubnium 𬭊
Sg seaborgium 𬭳
Bh bohrium 𬭛
Hs hassium 𬭶
Mt meitnerium 鿏
Ds darmstadtium 𫟼
Rg roentgenium 𬬭
Cn copernicium 鿔
Nh nihonium 鿭
Fl flerovium 鈇
Mc moscovium 镆
Lv livermorium 鉝
Ts tennessine 鿬
Og oganesson 鿫
"""
ELEMENTS = tuple(tuple(line.split()) for line in _ELEMENT_DATA.strip().splitlines())
# Explicit, manually maintained isotope presets, in atomic-number order. These
# are convenience choices (including radioactive nuclides), not atomic weights,
# measured masses, recommended charge states, or a complete nuclide database.
_DEFAULT_ISOTOPES = dict(
    enumerate((
        1,
        4,
        7,
        9,
        11,
        12,
        14,
        16,
        19,
        20,
        23,
        24,
        27,
        28,
        31,
        32,
        35,
        40,
        39,
        40,
        45,
        48,
        51,
        52,
        55,
        56,
        59,
        58,
        63,
        64,
        69,
        74,
        75,
        80,
        79,
        84,
        85,
        88,
        89,
        90,
        93,
        98,
        98,
        102,
        103,
        106,
        107,
        114,
        115,
        120,
        121,
        130,
        127,
        129,
        133,
        138,
        139,
        140,
        141,
        142,
        145,
        152,
        153,
        158,
        159,
        164,
        165,
        166,
        169,
        174,
        175,
        180,
        181,
        184,
        187,
        192,
        193,
        195,
        197,
        202,
        205,
        208,
        209,
        209,
        210,
        222,
        223,
        226,
        227,
        232,
        231,
        238,
        237,
        244,
        243,
        247,
        247,
        251,
        252,
        257,
        258,
        259,
        266,
        267,
        268,
        269,
        270,
        269,
        277,
        281,
        282,
        285,
        286,
        290,
        290,
        293,
        294,
        294,
    ), 1))
_ALIASES = {"aluminum": "aluminium", "cesium": "caesium", "sulphur": "sulfur"}
_SPECIAL = {
    "p": (1, 1, 1),
    "p+": (1, 1, 1),
    "proton": (1, 1, 1),
    "质子": (1, 1, 1),
    "e-": (0, -1, None),
    "electron": (0, -1, None),
    "电子": (0, -1, None),
    "e+": (0, 1, None),
    "positron": (0, 1, None),
    "正电子": (0, 1, None),
    "d": (2, 1, 1),
    "deuteron": (2, 1, 1),
    "氘": (2, 1, 1),
    "t": (3, 1, 1),
    "triton": (3, 1, 1),
    "氚": (3, 1, 1),
    "alpha": (4, 2, 2),
    "α": (4, 2, 2),
    "阿尔法": (4, 2, 2)
}

SPECIAL_PARTICLES = {
    # key: label, A, q, Z, mass key, searchable names
    "e-": ("电子 e−", 0, -1, None, "e", "electron 电子 e-"),
    "e+": ("正电子 e+", 0, 1, None, "e", "positron 正电子 e+"),
    "mu-": ("缪子 μ−", 0, -1, None, "mu", "muon 缪子 μ- mu-"),
    "mu+": ("反缪子 μ+", 0, 1, None, "mu", "antimuon 反缪子 μ+ mu+"),
    "tau-": ("τ 轻子 τ−", 0, -1, None, "tau", "tauon tau- τ-"),
    "tau+": ("反 τ 轻子 τ+", 0, 1, None, "tau", "antitau tau+ τ+"),
    "pi+": ("π+ 介子", 0, 1, None, "pi", "pion pi+ π+ π介子 pi介子"),
    "pi-": ("π− 介子", 0, -1, None, "pi", "negative-pion pi- π-"),
    "pi0": ("π0 介子", 0, 0, None, "pi0", "neutral-pion pi0 π0"),
    "K+": ("K+ 介子", 0, 1, None, "K", "kaon K+"),
    "K-": ("K− 介子", 0, -1, None, "K", "antikaon K-"),
    "K0": ("K0 介子", 0, 0, None, "K0", "neutral-kaon K0"),
    "n": ("中子 n", 1, 0, 0, "n", "neutron 中子 n"),
    "nbar": ("反中子 n̄", 1, 0, 0, "n", "antineutron 反中子 nbar"),
    "pbar": ("反质子 p̄", 1, -1, None, "p", "antiproton 反质子 pbar"),
}


@dataclass(frozen=True)
class ParticleSpec:
    mass_number: int
    charge_state: int
    atomic_number: int | None = None
    species: str = "ion"

    def __post_init__(self):
        a, q, z = self.mass_number, self.charge_state, self.atomic_number
        if type(a) is not int or a < 0 or type(q) is not int:
            raise ValueError("质量数 A 须为非负整数，电荷态 q 须为整数。")
        if z is not None and type(z) is not int:
            raise ValueError("质子数 Z 必须是整数或未指定。")
        if self.species == "ion" and a == 0 and z in (None, 0) and abs(q) == 1:
            object.__setattr__(self, "species", "e-" if q < 0 else "e+")
            object.__setattr__(self, "atomic_number", None)
        if self.species != "ion":
            entry = SPECIAL_PARTICLES.get(self.species)
            if entry is None or (a, q, self.atomic_number) != entry[1:4]:
                raise ValueError("粒子类型与 A、q、Z 不匹配；请重新选择粒子。")
            return
        if a == 0 or z is not None and not 1 <= z <= min(a, 118):
            raise ValueError("离子需要 1 ≤ Z ≤ min(A, 118)；Z=0 表示未指定元素。")
        elif q > (z if z is not None else a):
            raise ValueError("正电荷态不能大于质子数 Z（未指定元素时不能大于 A）。")

    @property
    def energy_divisor(self):
        """Nucleon count A for nuclei/baryons, one for leptons and mesons."""
        return self.mass_number if self.uses_nucleon_units else 1

    @property
    def uses_nucleon_units(self):
        return self.mass_number > 0

    @property
    def kinetic_energy_unit(self):
        return "AMeV" if self.uses_nucleon_units else "MeV"

    @property
    def normalized_rest_mass_ev_c2(self):
        """m0/A for A > 0, otherwise the complete particle mass (eV/c²)."""
        return self.rest_energy_ev / self.energy_divisor

    @property
    def mass_in_u(self):
        return self.rest_energy_ev / constant_mass("atomic mass constant")

    @property
    def rest_mass_per_u_ev_c2(self):
        """Numerical rest mass per u in eV/c²/u (equal to the u conversion)."""
        return constant_mass("atomic mass constant")

    @classmethod
    def from_species(cls, key):
        entry = SPECIAL_PARTICLES[key]
        return cls(*entry[1:4], species=key)

    @property
    def mass_record(self):
        if self.species != "ion":
            return special_mass(SPECIAL_PARTICLES[self.species][4])
        return ion_mass(self.mass_number, self.atomic_number, self.charge_state)

    @property
    def rest_energy_ev(self):
        return self.mass_record.energy_ev

    @property
    def label(self):
        if self.species != "ion":
            return SPECIAL_PARTICLES[self.species][0]
        if self.atomic_number is None:
            return f"未指定元素的离子 · A={self.mass_number}, q={self.charge_state:+d}"
        symbol, _, chinese = ELEMENTS[self.atomic_number - 1]
        charge = f"{abs(self.charge_state)}{'+' if self.charge_state > 0 else '−'}" if self.charge_state else "中性"
        return f"{symbol}-{self.mass_number}  {charge} · {chinese}"

    @property
    def mass_note(self):
        record = self.mass_record
        return record.note or ("Ek = K/A，使用当前电荷态的准确质量。" if self.uses_nucleon_units else "Ek 为单粒子动能，使用该粒子的准确质量。")


def _normalize(text):
    text = unicodedata.normalize("NFKC", text).strip().replace("−", "-").replace("^", "")
    return text.split(" · ")[0].strip().replace(" 中性", " 0+")


def _element(text):
    lower = _ALIASES.get(text.casefold(), text.casefold())
    for z, (symbol, english, chinese) in enumerate(ELEMENTS, 1):
        if lower in (symbol.casefold(), english, chinese):
            return z
    raise ValueError("未匹配元素；请从搜索建议中选择，或填写质子数 Z。")


def resolve_particle(text: str, mass_number=None, charge_state=None) -> ParticleSpec:
    """Resolve an exact name or isotope, never silently resolve fuzzy matches.

    Omitted A uses the element's explicit isotope preset; omitted q uses Z
    (fully stripped). Optional caller defaults override these conveniences,
    while explicit notation always wins. ``P`` is phosphorus; ``p`` is proton.
    """
    text = _normalize(text)
    for key, entry in SPECIAL_PARTICLES.items():
        if text == _normalize(entry[0]) or text != "N" and text.casefold() in [s.casefold() for s in entry[5].split()]:
            return ParticleSpec.from_species(key)
    if text != "P" and text.casefold() in _SPECIAL:
        return ParticleSpec(*_SPECIAL[text.casefold()])

    def spec(name, a=None, q=None):
        z = _element(name)
        return ParticleSpec(a if a is not None else mass_number if mass_number is not None else _DEFAULT_ISOTOPES[z],
                            q if q is not None else charge_state if charge_state is not None else z, z)

    # An initial mass number makes a following number unambiguously a charge.
    match = re.fullmatch(r"(\d+)\s*([A-Za-z]{1,3})(?:(\d*)\s*([+-]))?", text)
    if match:
        a, symbol, magnitude, sign = match.groups()
        q = (int(magnitude or 1) * (1 if sign == "+" else -1)) if sign else None
        return spec(symbol, int(a), q)
    # C-12, U-238 35+, carbon-12, 碳12, C12; suffix charge needs a separator.
    match = re.fullmatch(r"([^\d\s+-]+)\s*-?\s*(\d+)(?:\s+(\d*)\s*([+-]))?", text)
    if match:
        name, a, magnitude, sign = match.groups()
        q = (int(magnitude or 1) * (1 if sign == "+" else -1)) if sign else None
        return spec(name, int(a), q)
    # H-, C6+ (element with charge, mass from the preset unless supplied).
    match = re.fullmatch(r"([A-Za-z]{1,3}|[^\W\d_]+?)\s*(\d*)\s*([+-])", text)
    if match:
        name, magnitude, sign = match.groups()
        return spec(name, q=int(magnitude or 1) * (1 if sign == "+" else -1))
    return spec(text)


def search_particles(query: str) -> list[str]:
    """Chinese/name/symbol substring and typo suggestions, with explicit Z.

    Every suggestion specifies A/q/Z. Isotope presets are conveniences, not a
    complete nuclide table. Exact notations preserve their explicit A and q.
    """
    query = _normalize(query)
    term = re.sub(r"[\d\s+\-]", "", query).casefold()
    term = _ALIASES.get(term, term)
    result = []
    try:
        exact = resolve_particle(query)
    except ValueError:
        exact = None
    if exact is not None and exact.species != "ion":
        result.append((-2, SPECIAL_PARTICLES[exact.species][0]))
    for key, entry in SPECIAL_PARTICLES.items():
        aliases = entry[5].casefold()
        if (not query or query.casefold() in aliases
                or len(query) >= 3 and any(SequenceMatcher(None, query.casefold(), a).ratio() >= .7 for a in aliases.split())):
            if not any(name == entry[0] for _, name in result):
                result.append((0, entry[0]))
    for name in ("质子", "proton"):
        if query and query.casefold() in name:
            result.append((-1 if resolve_particle(name) == exact else 1, name))
    for z, (symbol, english, chinese) in enumerate(ELEMENTS, 1):
        aliases = (symbol.casefold(), english, chinese, str(z))
        if not query:
            score = 1 if z in _DEFAULT_ISOTOPES else 3
        elif query == str(z) or term in aliases:
            score = 0
        elif term and any(term in alias for alias in aliases):
            score = 1
        elif len(term) >= 3 and SequenceMatcher(None, term, english).ratio() >= .65:
            score = 2
        else:
            continue
        if exact is not None and exact.atomic_number == z:
            score = -1
        particle = ParticleSpec(_DEFAULT_ISOTOPES[z], z, z)
        if query:
            try:
                parsed = resolve_particle(query)
                if parsed.atomic_number == z:
                    particle = parsed
            except ValueError:
                pass
        name = f"{symbol}-{particle.mass_number} {abs(particle.charge_state)}{'+' if particle.charge_state > 0 else '-'}"
        result.append((score, f"{name} · {chinese} · {english} · Z={z}"))
    return [name for _, name in sorted(result, key=lambda item: item[0])][:30]
