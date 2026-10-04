# This template is appended to the generated module by export_dynamic_aperture.py.
# Reader, plotting, and export functions are provided by that module.


def _publish_outputs(products, targets):
    """Create all outputs exclusively; roll back this invocation on failure."""
    from contextlib import ExitStack
    import os
    import shutil

    created = {}
    try:
        with ExitStack() as stack:
            streams = []
            for path in targets:
                stream = stack.enter_context(path.open("xb"))
                created[path] = os.fstat(stream.fileno())
                streams.append(stream)
            for product, stream in zip(products, streams):
                with product.open("rb") as source:
                    shutil.copyfileobj(source, stream)
    except BaseException:
        for path, identity in created.items():
            try:
                # Preserve a replacement written by another process.
                if os.path.samestat(path.stat(), identity):
                    path.unlink()
            except OSError:
                pass
        raise


def main(argv=None):
    """Read a monitor and save a figure, without overwriting existing files."""
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="Plot dynamic aperture from one ParticleMonitor HDF5 or TFS file.")
    parser.add_argument("file", type=Path, help="Current single-file ParticleMonitor output (format version 2).")
    parser.add_argument("--turn", type=int, help="Inclusive simulation turn at this monitor; defaults to its requested final turn.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--dp", type=float, help="One exact initial dp group; defaults to the first group.")
    selection.add_argument("--dp-values", type=float, nargs="+", help="Initial dp groups to include with --overlay.")
    parser.add_argument("--overlay", action="store_true", help="Overlay boundaries for all dp groups, or the --dp-values subset.")
    parser.add_argument("--mode", choices=("status", "loss_turn"), default="status", help="Single-dp scatter coloring.")
    parser.add_argument("--output", type=Path, required=True, help="New figure path ending in .png, .pdf or .svg.")
    parser.add_argument("--data", type=Path, help="Optional new .npz or .csv result file; CSV also writes a .json metadata file.")
    args = parser.parse_args(argv)
    if args.turn is not None and args.turn < 0:
        parser.error("--turn must be nonnegative")
    if args.overlay and args.dp is not None:
        parser.error("--dp selects a single plot; use --dp-values with --overlay")
    if args.dp_values is not None and not args.overlay:
        parser.error("--dp-values requires --overlay")
    if args.overlay and args.mode != "status":
        parser.error("--mode loss_turn applies only to a single-dp plot")
    if args.output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        parser.error("--output must end in .png, .pdf or .svg")
    if args.data is not None and args.data.suffix.lower() not in {".npz", ".csv"}:
        parser.error("--data must end in .npz or .csv")
    targets = [args.output]
    if args.data is not None:
        targets.append(args.data)
        if args.data.suffix.lower() == ".csv":
            targets.append(args.data.with_suffix(".json"))
    resolved = [path.resolve() for path in targets]
    if len(set(resolved)) != len(resolved):
        parser.error("Output paths must be distinct")
    for path in targets:
        if path.exists():
            parser.error(f"Refusing to overwrite an existing file: {path}")
        if not path.parent.is_dir():
            parser.error(f"Output directory does not exist: {path.parent}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = None
    try:
        result = read_dynamic_aperture(args.file, requested_turn=args.turn)
        if args.overlay:
            ax = plot_dynamic_aperture_boundaries(result, dp_values=args.dp_values)
        else:
            ax = plot_dynamic_aperture(result, dp=args.dp, mode=args.mode)
        with tempfile.TemporaryDirectory(prefix="pass_da_") as directory:
            temporary = Path(directory)
            image_path = temporary / ("aperture" + args.output.suffix.lower())
            ax.figure.savefig(image_path, dpi=180)
            products = [image_path]
            if args.data is not None:
                data_path = temporary / ("results" + args.data.suffix.lower())
                export_dynamic_aperture(result, data_path)
                products.append(data_path)
                if args.data.suffix.lower() == ".csv":
                    products.append(data_path.with_suffix(".json"))
            # Exclusive creation also protects files appearing after preflight.
            _publish_outputs(products, targets)
        print(f"Analyzed monitor turn {result['metadata']['requested_turn']}; {len(result['particle_id'])} particles.")
        for path in targets:
            print(path.resolve())
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, f"Dynamic-aperture analysis failed: {error}\n")
    finally:
        if ax is not None:
            plt.close(ax.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
