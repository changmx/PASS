"""Pre-tracking SC weight and periodic integration-interval diagnostics."""
from collections import defaultdict
import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


def _interval_coverage(intervals, circumference, tolerance):
    """Sweep periodic intervals, retaining multiplicity for repeated coverage."""
    events = defaultdict(int)
    events[0.0] = events[circumference] = 0
    for start, length in intervals:
        laps, remainder = divmod(length, circumference)
        events[0.0] += int(laps)
        events[circumference] -= int(laps)
        if remainder <= tolerance:
            continue
        start %= circumference
        end = start + remainder
        if end <= circumference:
            events[start] += 1
            events[end] -= 1
        else:
            events[start] += 1
            events[circumference] -= 1
            events[0.0] += 1
            events[end - circumference] -= 1
    gaps, overlaps = [], []
    covered = 0.0
    count = 0
    previous = 0.0
    for position, delta in sorted(events.items()):
        if position - previous > tolerance:
            if count == 0:
                gaps.append([previous, position])
            else:
                covered += position - previous
                if count > 1:
                    overlaps.append({"start": previous, "end": position, "count": count})
        count += delta
        previous = position
    return covered, gaps, overlaps


def analyse_sc_coverage(commands, circumference, *, mode="full-ring", expected_length=None):
    """Analyse actual commands once per beam/pass, independent of bunch/turn count.

    Explicit SC weights without sc_start are counted but never assigned invented
    spatial intervals. Internal intervals come from the owning element body.
    """
    contributions = []
    internal_count = explicit_count = 0
    issues = []
    for command in commands:
        nodes = getattr(command, "_sc_nodes", {})
        if nodes:
            weights = [node_command.sc_length for _, node_command in nodes.values()]
            length = math.fsum(weights)
            internal_count += len(weights)
            if not math.isclose(length, command.length, rel_tol=1e-10, abs_tol=1e-12):
                issues.append(f"Element {command.cmd_name!r}: internal SC weights {length:g} m differ from body length {command.length:g} m")
            contributions.append({"name": command.cmd_name, "source": "internal", "length": length,
                                  "start": command.s - command.length, "extent": command.length})
        elif str(getattr(command, "cmd_type", "")).lower() == "spacecharge" and command.is_enabled:
            if command.sc_length == 0:
                continue
            explicit_count += 1
            contributions.append({"name": command.cmd_name, "source": "explicit", "length": command.sc_length,
                                  "start": getattr(command, "sc_start", None), "extent": command.sc_length})

    total = math.fsum(item["length"] for item in contributions)
    unknown = [item["name"] for item in contributions if item["start"] is None]
    report = {"mode": mode, "total_sc_length": total, "circumference": None,
              "target_length": None, "length_difference": None, "internal_kicks": internal_count,
              "explicit_kicks": explicit_count, "contributions": contributions,
              "unknown_intervals": unknown, "covered_length": None, "uncovered_intervals": [],
              "overlap_intervals": [], "issues": issues, "status": "ok"}
    try:
        circumference = float(circumference)
        if not math.isfinite(circumference) or circumference <= 0:
            raise ValueError
    except (TypeError, ValueError):
        issues.append("A finite positive Circumference (m) is required for periodic SC coverage validation")
        report["status"] = "incomplete"
        return report

    report["circumference"] = circumference
    tolerance = max(1e-12, 1e-10 * circumference)
    report["tolerance"] = tolerance
    target = circumference if mode == "full-ring" else expected_length
    report["target_length"] = target
    if target is not None:
        report["length_difference"] = total - target
        if abs(total - target) > tolerance:
            issues.append(f"SC total length {total:g} m differs from target {target:g} m by {total-target:+g} m")

    intervals = [(item["start"], item["extent"]) for item in contributions if item["start"] is not None]
    covered, gaps, overlaps = _interval_coverage(intervals, circumference, tolerance)
    report.update(covered_length=covered, uncovered_intervals=gaps, overlap_intervals=overlaps)
    if unknown:
        issues.append(f"Explicit SC commands lack SC start (m): {unknown}; total weights were checked, spatial coverage is incomplete")
    if overlaps:
        issues.append(f"SC integration intervals overlap in {len(overlaps)} region(s)")
    if mode == "full-ring" and gaps:
        if unknown:
            issues.append("Known SC intervals leave regions uncovered; commands with unspecified intervals may or may not cover them")
        else:
            issues.append(f"SC integration intervals leave {len(gaps)} gap(s) totaling {math.fsum(b-a for a,b in gaps):g} m")
    report["status"] = "incomplete" if unknown else ("mismatch" if issues else "ok")
    return report


def validate_sc_coverage(sim, sequences):
    """Report every enabled beam, then optionally fail before any tracking."""
    settings = getattr(sim.cfg, "space_charge", [])
    reports = {}
    failures = []
    if not any(config.enabled for config in settings):
        sim.space_charge_coverage = reports
        return reports
    for sequence in sequences:
        beam_id = sequence.beam_id
        if beam_id >= len(settings) or not settings[beam_id].enabled:
            continue
        config = settings[beam_id]
        if config.coverage_check == "off":
            logger.info("Beam %d SC coverage check: off", beam_id)
            reports[beam_id] = {"beam_id": beam_id, "status": "disabled"}
            continue
        data = sim.cfg.input_data[beam_id]
        report = analyse_sc_coverage(sequence.cmds, data.get("circumference (m)"),
                                     mode=config.coverage_mode, expected_length=config.expected_sc_length)
        report["beam_id"] = beam_id
        report["policy"] = config.coverage_check
        reports[beam_id] = report
        logger.info("Beam %d SC coverage: mode=%s, total=%g m, target=%s m, circumference=%s m, "
                    "internal kicks=%d, explicit kicks=%d, status=%s", beam_id, report["mode"],
                    report["total_sc_length"], report["target_length"], report["circumference"],
                    report["internal_kicks"], report["explicit_kicks"], report["status"])
        for key in ("uncovered_intervals", "overlap_intervals"):
            regions = report[key]
            if regions:
                logger.info("Beam %d SC %s (m), first 10 of %d: %s", beam_id, key, len(regions), regions[:10])
        for issue in report["issues"]:
            logger.warning("Beam %d SC coverage: %s", beam_id, issue)
        output = getattr(sim.cfg, "output_dir_space_charge", None)
        if output:
            path = Path(output) / f"coverage_beam{beam_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        if config.coverage_check == "error" and report["issues"]:
            failures.append(f"beam {beam_id}: " + "; ".join(report["issues"]))
    sim.space_charge_coverage = reports
    if failures:
        raise ValueError("Space-charge coverage validation failed before tracking: " + " | ".join(failures))
    return reports
