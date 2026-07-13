"""End-to-end test: generate input JSON → engine reads it.

Tests the full pipeline:
    schema → Sequence → JSON writer → Config.load_input → CommandSequence
"""

import json
import sys
import os
import tempfile
from pathlib import Path

# Ensure PASS is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PASS.para.api import generate_input
from PASS.para.schema.main import MainConfig
from PASS.para.schema.bunch import BunchConfig, InjectionItem
from PASS.para.schema.sequence import Sequence
from PASS.para.schema.monitors import StatMonitor
from PASS.para.readers.smooth_approx import generate_smooth_twiss

# ---- 1. Build main config ----
main = MainConfig(
    beam_name="proton",
    num_proton=1,
    num_neutron=0,
    num_electron=1,
    gamma_t=4.8,
    circumference=251.327,
    num_turns=5,
    backend="cpu",
    output_dir="./output",
    is_plot=False,
)
print(f"1. MainConfig: beam={main.beam_name}, turns={main.num_turns}, C={main.circumference}")

# ---- 2. Build bunch ----
bunch = BunchConfig(
    kinetic_energy=45e6,
    num_real_particles=int(1e11),
    num_macro_particles=int(1e5),
    beta_x=0.5,
    beta_y=0.5,
    alpha_x=-2.614303952,
    alpha_y=1.57442348,
    emit_x=200e-6,
    emit_y=100e-6,
    sigma_z=30,
    dp=0.005,
    dist_trans="gaussian",
    dist_longi="matchz",
    rf_voltage=100e3,
    rf_phase=0.5235987755982988,
    save_init_dist=True,
)
print(f"2. BunchConfig: Ek={bunch.kinetic_energy:.2e}, Np={bunch.num_macro_particles}")

# ---- 3. Generate smooth twiss ----
items, circum = generate_smooth_twiss(
    circumference=main.circumference,
    qx=4.8,
    qy=4.4,
    num_points=50,
    muz=0.001,
    longitudinal_transfer="off",
)
print(f"3. Smooth twiss: {len(items)} points, C={circum}")

# ---- 4. Assemble sequence ----
seq = Sequence()
seq.add("injection", InjectionItem(s=0.0, bunches=[bunch]))
for i, item in enumerate(items):
    seq.add(f"twiss_{i:04d}", item)
seq.add("stat1", StatMonitor(s=0.0))
print(f"4. Sequence: {len(seq)} items")

# ---- 5. Generate JSON ----
output_path = os.path.join(tempfile.gettempdir(), "pass_test_beam0.json")
generate_input(main, seq, output_path)
print(f"5. JSON written: {output_path}")

# ---- 6. Verify JSON content ----
with open(output_path, "r") as f:
    data = json.load(f)

print(f"\n6. JSON verification:")
print(f"   Root keys: {sorted(data.keys())}")
print(f"   Sequence items: {len(data['Sequence'])}")
print(f"   First sequence key: {list(data['Sequence'].keys())[0]}")
print(f"   Has injection: {'injection' in data['Sequence']}")
print(f"   Injection has bunch0: {'bunch0' in data['Sequence']['injection']}")

# ---- 7. Engine reads it ----
print(f"\n7. Engine compatibility test:")

from PASS.core.config import Config
from PASS.utils.helper import convert_keys_to_lower

cfg = Config()
cfg.load_input(output_path)
print(f"   Config loaded: num_turn={cfg.num_turn}, backend={cfg.backend}")
print(f"   num_bunch={cfg.num_bunch}, beam_name={cfg.beam_name}")

from PASS.core.sequence import CommandSequence
from PASS.core.simulation import Simulation
from PASS.core.beam import Beam
from PASS.core.state import State

beams = [Beam(output_path, cfg)]
state = State()
sim = Simulation(cfg, beams, state)
print(f"   Beam created: {beams[0].num_bunch} bunch(es), Np_total={beams[0].Np_total}")

seq_obj = CommandSequence(output_path, beam_id=0, sim=sim)
seq_obj.sort()
print(f"   CommandSequence: {seq_obj.num_cmd} commands")
print(f"   First 5 commands:")
for cmd in seq_obj.cmds[:5]:
    print(f"     {cmd.cmd_type} @ s={cmd.s:.4f}")
print(f"   Last 3 commands:")
for cmd in seq_obj.cmds[-3:]:
    print(f"     {cmd.cmd_type} @ s={cmd.s:.4f}")

print(f"\n=== END-TO-END TEST PASSED ===")
