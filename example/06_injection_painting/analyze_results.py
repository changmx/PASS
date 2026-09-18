"""Plot injection history and batch identities from existing DistMonitor files."""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tfs
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_snapshot(path):
    if path.suffix == ".h5":
        with h5py.File(path) as stream:
            return pd.DataFrame({k: stream[k][:] for k in stream}), dict(stream.attrs)
    frame = tfs.read(path)
    return pd.DataFrame(frame), dict(frame.headers)


def write_gallery(output):
    """Write a self-contained navigator for the local scientific PNG files."""
    output = Path(output)
    manifest_path = output / "plot_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collections = {
        "Injection evolution": [{
            "image": Path(path).name,
            "label": f"Turn {int(Path(path).stem.rsplit('_', 1)[1])}"
        } for path in manifest["snapshots"]]
    }
    for label, pattern in (("Injection batches", "injection_complete_batches_*.png"), ("Full density", "injection_complete_density.png"),
                           ("Batch population", "batch_population_at_injection_complete.png"), ("CISP comparison", "statistics_comparison_*.png")):
        files = sorted(output.glob(pattern))
        if files:
            collections[label] = [{"image": path.name, "label": path.stem.replace("_", " ")} for path in files]
    payload = json.dumps(collections).replace("<", "\\u003c")
    complete = manifest.get("injection_complete_turn")
    status = (f"Injection complete at turn {complete}: {manifest['batches']} batches."
              if complete is not None else "Injection is incomplete in these snapshots.")
    page = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Injection painting results</title>
<style>
body{font:16px system-ui,sans-serif;color:#182f42;background:#f1f5f8;margin:0}
main{max-width:1550px;margin:auto;padding:24px}h1{font-size:27px;margin:0 0 8px}
p{line-height:1.5}nav{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:white;padding:14px;border-radius:10px}
button,select{font:inherit;padding:7px 12px;border:1px solid #b5c4cf;border-radius:5px;background:white}
button{cursor:pointer}button:disabled{cursor:default;opacity:.5}input{flex:1;min-width:150px}
figure{margin:18px 0;background:white;border-radius:10px;padding:8px}img{display:block;width:100%;height:auto}
figcaption{padding:10px;font-weight:600}a{color:#1761a3}small{color:#4f6474}
</style><main>
<h1>Injection painting results</h1><p>__STATUS__ Turn numbers start at zero.</p>
<nav aria-label="Figure controls"><label>View <select id="view"></select></label>
<button id="previous" aria-label="Previous frame">Previous</button><button id="play">Play</button>
<button id="next" aria-label="Next frame">Next</button>
<input id="frame" aria-label="Frame" type="range" min="0" value="0"><output id="counter"></output></nav>
<figure><figcaption id="caption"></figcaption><img id="plot" alt="Injection phase-space figure"></figure>
<p>Colors identify the highlighted injection batches; gray shows other survivors. Scatter sampling affects display only.
All saved particle distributions retain every born particle, including losses.</p>
<p><a href="snapshot_counts.csv">Snapshot counts (CSV)</a> · <a href="plot_manifest.json">Figure manifest</a></p>
<small>Use the arrow keys to move between frames. The PNG figures and this page work together without a network connection.</small>
</main><script>
const collections=__COLLECTIONS__;
const view=document.getElementById('view'), slider=document.getElementById('frame');
const plot=document.getElementById('plot'), caption=document.getElementById('caption');
const previous=document.getElementById('previous'), next=document.getElementById('next'), play=document.getElementById('play');
let timer=null;
for(const label of Object.keys(collections)){const option=document.createElement('option');option.textContent=label;view.appendChild(option);}
function stop(){if(timer!==null){clearInterval(timer);timer=null;}play.textContent='Play';}
function render(){const frames=collections[view.value], i=Number(slider.value), item=frames[i];
  plot.src=item.image;plot.alt=item.label;caption.textContent=item.label;
  document.getElementById('counter').textContent=`${i+1} / ${frames.length}`;
  previous.disabled=i===0;next.disabled=i===frames.length-1;play.disabled=frames.length<2;}
function changeView(){stop();slider.max=collections[view.value].length-1;slider.value=0;render();}
function step(delta){stop();slider.value=Math.max(0,Math.min(Number(slider.max),Number(slider.value)+delta));render();}
view.addEventListener('change',changeView);slider.addEventListener('input',()=>{stop();render();});
previous.addEventListener('click',()=>step(-1));next.addEventListener('click',()=>step(1));
play.addEventListener('click',()=>{if(timer!==null){stop();return;}play.textContent='Pause';
  timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%(Number(slider.max)+1);render();},900);});
document.addEventListener('keydown',event=>{if(event.target.tagName==='SELECT'||event.target.tagName==='INPUT')return;
  if(event.key==='ArrowRight'){event.preventDefault();step(1);}if(event.key==='ArrowLeft'){event.preventDefault();step(-1);}});
changeView();
</script></html>"""
    (output / "index.html").write_text(page.replace("__STATUS__", status).replace("__COLLECTIONS__", payload), encoding="utf-8")


def analyse(directory, output=None, max_points_per_batch=2500, beam_id=0, bunch_id=0):
    if max_points_per_batch < 1:
        raise ValueError("max_points_per_batch must be positive")
    directory = Path(directory)
    output = Path(output) if output else directory / "painting_plots"
    output.mkdir(parents=True, exist_ok=True)
    pattern = f"*_dist_beam{beam_id}_bunch{bunch_id}_*after_injection*"
    paths = list(directory.rglob(pattern + ".h5")) + list(directory.rglob(pattern + ".tfs"))
    if not paths:
        raise FileNotFoundError("No after_injection DistMonitor snapshots found")
    snapshots, ranges, counts, complete_snapshots = [], {}, [], []
    fields = ("x", "px", "y", "py")
    for path in paths:
        frame, metadata = read_snapshot(path)
        for key in ("particle_id", "injection_turn", "injection_batch"):
            if key not in frame:
                raise ValueError("Snapshots require Include injection metadata=true")
        if frame.particle_id.duplicated().any() or (frame.injection_batch < 0).any() or (frame.injection_turn < 0).any():
            raise ValueError("Snapshots require unique IDs and valid birth metadata for every saved particle")
        alive = frame[frame.tag > 0]
        for key in fields:
            if len(alive):
                lo, hi = float(alive[key].min() * 1000), float(alive[key].max() * 1000)
                old = ranges.get(key, (lo, hi))
                ranges[key] = (min(old[0], lo), max(old[1], hi))
        turn = int(metadata["Turn"])
        snapshots.append((turn, path))
        if int(metadata.get("NumPending", -1)) == 0 and len(frame):
            complete_snapshots.append((turn, path))
        counts.append({
            "turn": turn,
            "alive": len(alive),
            "lost": int((frame.tag < 0).sum()),
            "injected": len(frame),
            "pending": int(metadata.get("NumPending", 0))
        })
    snapshots.sort()
    if len({turn for turn, _ in snapshots}) != len(snapshots):
        raise ValueError("Select one run and one bunch; duplicate snapshot turns found")
    for key in fields:
        lo, hi = ranges.get(key, (-1., 1.))
        pad = max(.04 * (hi - lo), .001)
        ranges[key] = (lo - pad, hi + pad)
    palette = plt.get_cmap("tab10")
    panels = [("x", "px"), ("y", "py"), ("x", "y")]
    labels = {"x": "x (mm)", "y": "y (mm)", "px": r"$p_x=P_x/P_0$ ($10^{-3}$)", "py": r"$p_y=P_y/P_0$ ($10^{-3}$)"}
    summary = []

    def sample(frame, limit):
        frame = frame.sort_values("particle_id")
        if len(frame) > limit:
            # Fixed ID order makes display selection reproducible.
            frame = frame.iloc[np.linspace(0, len(frame) - 1, limit, dtype=int)]
        return frame

    def plot(frame, title, path, batches=None):
        alive = frame[frame.tag > 0]
        other_batches = alive if batches is None else alive[~alive.injection_batch.isin(batches)]
        background = sample(other_batches, 30000)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
        for ax, (x, y) in zip(axes, panels):
            if batches is None:
                if len(alive):
                    ax.hist2d(alive[x] * 1000, alive[y] * 1000, bins=150, range=[ranges[x], ranges[y]], cmap="viridis", cmin=1)
            else:
                ax.scatter(background[x] * 1000, background[y] * 1000, s=1, c="0.83", rasterized=True)
                for batch in batches:
                    part = sample(alive[alive.injection_batch == batch], max_points_per_batch)
                    ax.scatter(part[x] * 1000,
                               part[y] * 1000,
                               s=3,
                               alpha=.7,
                               color=palette(int(batch) % 10),
                               label=f"Batch {int(batch)+1} ({len(alive[alive.injection_batch == batch]):,})")
            ax.set(xlabel=labels[x], ylabel=labels[y], xlim=ranges[x], ylim=ranges[y])
            ax.grid(alpha=.15)
        if batches is not None:
            axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
        fig.suptitle(title, fontsize=12)
        fig.savefig(path, dpi=150)
        plt.close(fig)

    for turn, path in snapshots:
        frame, metadata = read_snapshot(path)
        newest = int(frame.injection_batch.max()) if len(frame) else -1
        image = output / f"injection_evolution_turn_{turn:05d}.png"
        plot(frame,
             f"After injection point | turn {turn} | {int((frame.tag>0).sum()):,} survivors | newest batch {newest+1}",
             image,
             batches=[newest] if newest >= 0 else [])
        summary.append(str(image.resolve()))
    # Keep both injection-completion and later evolution frames. Group pages
    # refer to the frame immediately after the last batch is born.
    completion = min(complete_snapshots) if complete_snapshots else None
    last_batch = None
    if completion is not None:
        frame, metadata = read_snapshot(completion[1])
        last_batch = int(frame.injection_batch.max())
        population = frame.groupby("injection_batch").tag.agg(injected="size",
                                                              alive=lambda values: int((values > 0).sum()),
                                                              lost=lambda values: int((values < 0).sum())).reset_index()
        population["survival_fraction"] = population.alive / population.injected
        population.to_csv(output / "batch_population_at_injection_complete.csv", index=False)
        fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
        batch_numbers = population.injection_batch + 1
        ax.bar(batch_numbers, population.alive, color=[palette(int(i) % 10) for i in population.injection_batch], label="Alive")
        ax.bar(batch_numbers, population.lost, bottom=population.alive, color="0.8", label="Lost")
        ax.set(xlabel="Injection batch (one-based)",
               ylabel="Macro-particles",
               title=f"Batch population immediately after injection completes | turn {completion[0]}")
        ax.legend()
        fig.savefig(output / "batch_population_at_injection_complete.png", dpi=150)
        plt.close(fig)
        plot(frame, f"Injection complete | turn {completion[0]} | full surviving distribution", output / "injection_complete_density.png")
        for first in range(0, last_batch + 1, 10):
            stop = min(first + 10, last_batch + 1)
            plot(frame, f"Injection complete | turn {completion[0]} | batches {first+1}-{stop}",
                 output / f"injection_complete_batches_{first+1:02d}_{stop:02d}.png", range(first, stop))
    pd.DataFrame(counts).sort_values("turn").to_csv(output / "snapshot_counts.csv", index=False)
    (output / "plot_manifest.json").write_text(json.dumps(
        {
            "snapshots": summary,
            "injection_complete_turn": completion[0] if completion else None,
            "batches": last_batch + 1 if last_batch is not None else None,
            "scatter_sampling_only": True,
            "maximum_displayed_per_batch": max_points_per_batch
        },
        indent=2),
                                               encoding="utf-8")
    write_gallery(output)
    return output


def compare_statistics(pass_csv, cisp_csv, output, note="Verify that the two runs use the same physical settings"):
    """Compare only caller-selected runs at the same physical monitor."""
    measured, reference = pd.read_csv(pass_csv), pd.read_csv(cisp_csv)
    mapping = {
        "numAlive": ("particle_num", 1.),
        "xAverage": ("x_center(m)", 1.),
        "yAverage": ("y_center(m)", 1.),
        "xEmittance": ("emit_x(pi*m*rad)", 1.),
        "yEmittance": ("emit_y(pi*m*rad)", 1.)
    }
    # PASS reference energy is eV/u; CISP statistics report MeV/u.
    for key, other, factor in (("sigmaZ", "sigma_z(m)", 1.), ("sigmadp", "sigma_dp", 1.), ("Ek", "std_ke(MeV/u)", 1e6)):
        if key in measured and other in reference:
            mapping[key] = (other, factor)
    for label, frame in (("PASS", measured), ("CISP", reference)):
        if frame.turn.duplicated().any():
            raise ValueError(f"{label} statistics have duplicate turns; select one monitor/run")
    merged = measured.merge(reference, on="turn", suffixes=("_pass", "_cisp"), validate="one_to_one")
    if len(merged) != len(measured):
        raise ValueError("CISP statistics do not cover all PASS turns")
    result = pd.DataFrame({"turn": merged.turn})
    for key, (other, factor) in mapping.items():
        result[key + "_PASS"] = merged[key]
        result[key + "_CISP"] = merged[other] * factor
        result[key + "_difference"] = merged[key] - merged[other] * factor
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    rows = (len(mapping) + 3) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(13, 3.5 * rows), layout="constrained")
    units = {
        "numAlive": (1e-6, "Survivors (million)"),
        "xAverage": (1e3, "Mean x (mm)"),
        "yAverage": (1e3, "Mean y (mm)"),
        "xEmittance": (1e6, r"RMS emittance x ($\mu$m)"),
        "yEmittance": (1e6, r"RMS emittance y ($\mu$m)"),
        "sigmaZ": (1., "RMS bunch length (m)"),
        "sigmadp": (1e3, r"RMS momentum spread ($10^{-3}$)"),
        "Ek": (1e-6, "Reference kinetic energy (MeV/u)")
    }
    for ax, key in zip(axes.flat, mapping):
        scale, label = units[key]
        ax.plot(result.turn, result[key + "_PASS"] * scale, label="PASS")
        ax.plot(result.turn, result[key + "_CISP"] * scale, "--", label="CISP")
        ax.set(xlabel="Turn (zero-based)", ylabel=label)
        ax.grid(alpha=.2)
        ax.legend()
    for ax in list(axes.flat)[len(mapping):]:
        ax.axis("off")
    import textwrap
    axes.flat[-1].text(0, .9, textwrap.fill(note, 43), va="top", fontsize=10, transform=axes.flat[-1].transAxes)
    fig.suptitle("Statistics at the same physical monitor")
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)
    import hashlib
    output.with_suffix(".json").write_text(json.dumps(
        {
            "PASS_file": str(Path(pass_csv).resolve()),
            "CISP_file": str(Path(cisp_csv).resolve()),
            "CISP_sha256": hashlib.sha256(Path(cisp_csv).read_bytes()).hexdigest(),
            "note": note,
            "matched_turns": len(result),
            "CISP_column_conversion": {
                key: {
                    "column": other,
                    "factor_to_PASS_units": factor
                }
                for key, (other, factor) in mapping.items()
            },
            "numerical_agreement_certified": False
        },
        indent=2),
                                           encoding="utf-8")
    write_gallery(output.parent)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="One completed PASS run directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-points-per-batch", type=int, default=2500)
    parser.add_argument("--beam-id", type=int, default=0)
    parser.add_argument("--bunch-id", type=int, default=0)
    parser.add_argument("--cisp-pm2", type=Path, help="CISP bunch CSV at the injection point")
    parser.add_argument("--cisp-pm1", type=Path, help="CISP bunch CSV after B4")
    parser.add_argument("--comparison-note", default="Verify that the two runs use the same physical settings")
    args = parser.parse_args()
    output = analyse(args.directory, args.output, args.max_points_per_batch, args.beam_id, args.bunch_id)
    for label, position, reference in (("pm2", "0.0000", args.cisp_pm2), ("pm1", "1.9000", args.cisp_pm1)):
        if reference:
            paths = list(args.directory.glob(f"*_stat_beam{args.beam_id}_bunch{args.bunch_id}_*_s_{position}.csv"))
            if len(paths) != 1:
                raise ValueError(f"Expected one PASS statistics file at s={position}, found {len(paths)}")
            compare_statistics(paths[0], reference, output / f"statistics_comparison_{label}.csv", args.comparison_note)
    print(output)
