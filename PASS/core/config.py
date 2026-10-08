from dataclasses import dataclass, field
from copy import deepcopy
from typing import Literal
from pathlib import Path
import json
import os
import sys
import socket
import platform
import logging

import numpy as np

from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.helper import convert_keys_to_lower
from PASS.utils.program import LinearProgram
from PASS.utils.input_snapshot import create_run_directory, json_bytes, resolve_output_base

logger = logging.getLogger(__name__)


@dataclass
class Config:

    num_beam: int = 0
    beam_name: list[str] = field(default_factory=list)
    harmonic_number: list[int] = field(default_factory=list)
    num_turn: int = 0
    num_collision: int = 0
    use_cpu: bool = False
    use_gpu: bool = True
    particle_precision: str = "float64"
    num_gpu: int = 0
    gpu_id: list[int] = field(default_factory=list)
    is_plot: bool = False
    input_path: list[str] = field(default_factory=list)
    input_data: list[dict] = field(default_factory=list)
    space_charge: list = field(default_factory=list)
    space_charge_configuration_counts: list[int] = field(default_factory=list)
    electron_cloud: list = field(default_factory=list)
    intrabeam_scattering: list = field(default_factory=list)
    beam_beam_enabled: bool = False
    beam_beam_configurations: dict = field(default_factory=dict)
    timing: dict = field(default_factory=lambda: {
        "mode": "command",
        "log_interval": 10,
        "warmup_turns": 1,
        "include_io": True,
    })
    output_interval: int = 100
    output_ymd: str = ""
    output_hms: str = ""
    output_dir: str = ""
    output_dir_log: str = ""
    output_dir_stat: str = ""
    output_dir_para: str = ""
    output_dir_dist: str = ""
    output_dir_tuneSpread: str = ""
    output_dir_chargeDensity: str = ""
    output_dir_space_charge: str = ""
    output_dir_plot: str = ""
    output_dir_particle: str = ""
    output_dir_slice: str = ""
    output_dir_slowExt_particle: str = ""

    def load_input(self,
                   beam0_path: str,
                   beam1_path: str | None = None,
                   *,
                   flat_output: bool = False,
                   _run_directory: str | Path | None = None) -> None:
        """Load inputs; optionally write a caller-managed run into one directory.

        ``flat_output`` is a runtime option for isolated verification workflows.
        The caller must provide a separate output directory for every run.
        Normal application runs retain the dated output layout.
        Internal launchers pass _run_directory to reuse the directory reserved
        for the input archive, with the normal result filenames and subdirectories.
        """
        if flat_output and _run_directory is not None:
            raise ValueError("A prepared dated run directory cannot be combined with flat_output")
        self.flat_output = flat_output
        self.beam_name.clear()
        self.harmonic_number.clear()
        self.input_path.clear()
        self.input_data.clear()
        self.space_charge.clear()
        self.space_charge_configuration_counts.clear()
        self.electron_cloud.clear()
        self.intrabeam_scattering.clear()

        path0 = Path(beam0_path)
        if not path0.exists():
            raise FileNotFoundError(f"Input beam0 file not found: {path0}")
        with open(path0, 'r', encoding='utf-8-sig') as f:
            raw0 = json.load(f)
            from PASS.validation.files import resolve_input_paths
            resolve_input_paths(raw0, path0.resolve().parent)
            parameters0 = deepcopy(raw0)
            from PASS.para.schema.wake_field import expand_wake_configurations
            expand_wake_configurations(raw0)
            space_charge0, space_charge_count0 = self._load_space_charge(raw0)
            electron_cloud0 = self._load_electron_cloud(raw0)
            intrabeam_scattering0 = self._load_intrabeam_scattering(raw0)
            data0 = convert_keys_to_lower(raw0)
            data0["electron cloud"] = self._electron_cloud_engine_data(electron_cloud0)
            data0["intrabeam scattering"] = self._intrabeam_scattering_engine_data(intrabeam_scattering0)

        if beam1_path is not None:
            path1 = Path(beam1_path)
            if not path1.exists():
                raise FileNotFoundError(f"Input beam1 file not found: {path1}")
            with open(path1, 'r', encoding='utf-8-sig') as f:
                raw1 = json.load(f)
                resolve_input_paths(raw1, path1.resolve().parent)
                parameters1 = deepcopy(raw1)
                expand_wake_configurations(raw1)
                space_charge1, space_charge_count1 = self._load_space_charge(raw1)
                electron_cloud1 = self._load_electron_cloud(raw1)
                intrabeam_scattering1 = self._load_intrabeam_scattering(raw1)
                data1 = convert_keys_to_lower(raw1)
                data1["electron cloud"] = self._electron_cloud_engine_data(electron_cloud1)
                data1["intrabeam scattering"] = self._intrabeam_scattering_engine_data(intrabeam_scattering1)

        from PASS.commands.collision.config import load_beam_beam
        self.beam_beam_enabled, self.beam_beam_configurations = load_beam_beam([raw0] if beam1_path is None else [raw0, raw1])
        from PASS.validation.relations import find_slice_usage_conflicts
        for beam_id, data in enumerate([raw0] if beam1_path is None else [raw0, raw1]):
            conflicts = find_slice_usage_conflicts(data,
                                                   beam_id=beam_id,
                                                   beam_beam_configurations=self.beam_beam_configurations if self.beam_beam_enabled else None)
            if conflicts:
                raise ValueError(f"Beam {beam_id}: {conflicts[0][1]}")

        if beam1_path is None:
            self.num_beam = 1
        else:
            self.num_beam = 2

        if self.num_beam == 1:
            self.beam_name.append(data0.get("beam name"))
            self.input_path.append(beam0_path)
            self.input_data.append(data0)
            self.space_charge.append(space_charge0)
            self.space_charge_configuration_counts.append(space_charge_count0)
            self.electron_cloud.append(electron_cloud0)
            self.intrabeam_scattering.append(intrabeam_scattering0)
            h0 = int(data0["sequence"]["injection"]["harmonic number"])
            self.harmonic_number.append(h0)
        else:
            self.beam_name.append(data0.get("beam name"))
            self.beam_name.append(data1.get("beam name"))
            self.input_path.append(beam0_path)
            self.input_path.append(beam1_path)
            self.input_data.append(data0)
            self.input_data.append(data1)
            self.space_charge.append(space_charge0)
            self.space_charge.append(space_charge1)
            self.space_charge_configuration_counts.append(space_charge_count0)
            self.space_charge_configuration_counts.append(space_charge_count1)
            self.electron_cloud.extend((electron_cloud0, electron_cloud1))
            self.intrabeam_scattering.extend((intrabeam_scattering0, intrabeam_scattering1))
            h0 = int(data0["sequence"]["injection"]["harmonic number"])
            h1 = int(data1["sequence"]["injection"]["harmonic number"])
            self.harmonic_number.append(h0)
            self.harmonic_number.append(h1)

        self.num_turn = data0.get("number of turns", 0)
        self.timing = self._load_timing(data0.get("timing"))
        # self.num_collision = data0.get(["Number of collisions"])
        self.backend = data0.get("backend (gpu/cpu)", "cpu").lower()
        if self.backend == "gpu":
            self.use_gpu = True
            self.use_cpu = False
        elif self.backend == "cpu":
            self.use_gpu = False
            self.use_cpu = True
        else:
            raise ValueError(f"The backend should be cpu or gpu, but now is {self.backend}")

        self.particle_precision = data0.get("particle precision", "float64").lower()
        if self.particle_precision not in {"float32", "float64"}:
            self.particle_precision = "float64"
            logger.warning(f"Particle Precision must be 'float32' or 'float64', but got "
                           f"{self.particle_precision!r}. Defaulting to 'float64'.")
        if self.num_beam == 2:
            beam1_precision = data1.get("particle precision", "float64").lower()
            if beam1_precision != self.particle_precision:
                logger.warning(f"Both beam input files must use the same Particle Precision; "
                               f"got {self.particle_precision!r} and {beam1_precision!r}. "
                               f"Defaulting to 'float64'.")
                self.particle_precision = "float64"

        if self.use_gpu:
            self.num_gpu = int(data0.get("number of gpu devices", 1))
            configured_gpu_id = data0.get("device id", [0])
            if isinstance(configured_gpu_id, int):
                configured_gpu_id = [configured_gpu_id]
            self.gpu_id = [int(device_id) for device_id in configured_gpu_id]
            if not self.gpu_id:
                raise ValueError("At least one GPU device id must be configured")
            if self.num_gpu != len(self.gpu_id):
                logger.warning(
                    "Number of GPU devices (%s) does not match Device Id length (%s); "
                    "using the first configured device for this single-process run.",
                    self.num_gpu,
                    len(self.gpu_id),
                )

        self.is_plot = data0.get("is plot figure")

        output_base = resolve_output_base(data0.get("output directory"), beam0_path)
        parameters = [parameters0] if beam1_path is None else [parameters0, parameters1]
        for data in parameters:
            output_key = next((key for key in data if key.casefold() == "output directory"), "Output directory")
            data[output_key] = str(output_base)

        if flat_output:
            self.output_ymd = ""
            self.output_hms = "run"
            for name in self.__dataclass_fields__:
                if name == "output_dir" or name.startswith("output_dir_"):
                    setattr(self, name, str(output_base))
            Path(output_base).mkdir(parents=True, exist_ok=True)
            for index, data in enumerate(parameters):
                destination = Path(output_base) / f"run_beam{index}_snapshot.json"
                if destination.exists():
                    raise FileExistsError(f"Flat run snapshot already exists: {destination}")
                self._write_parameter_snapshot(destination, data)
            return

        if _run_directory is None:
            results = create_run_directory(output_base)
        else:
            results = Path(_run_directory).resolve()
            if len(results.relative_to(output_base).parts) != 2 or not results.is_dir():
                raise ValueError("The prepared run directory must exist directly beneath the output root's date directory")
        self.output_ymd = results.parent.name
        self.output_hms = results.name
        self.output_dir = str(results)
        self.output_dir_log = str(results)
        self.output_dir_stat = str(results)
        self.output_dir_para = str(results)
        self.output_dir_dist = str(results / "distribution")
        self.output_dir_tuneSpread = str(results / "tuneSpread")
        self.output_dir_chargeDensity = str(results / "chargeDensity")
        self.output_dir_space_charge = str(results / "space_charge")
        self.output_dir_plot = str(results / "plot")
        self.output_dir_particle = str(results / "particle")
        self.output_dir_slice = str(results / "slice")
        self.output_dir_slowExt_particle = str(results / "slowExt_particle")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir_log).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir_stat).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir_para).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir_dist).mkdir(parents=True, exist_ok=True)

        for index, data in enumerate(parameters):
            self._write_parameter_snapshot(Path(self.output_dir_para) / f"{self.output_hms}_beam{index}.json", data)

    @staticmethod
    def _write_parameter_snapshot(path, data):
        """Save loaded parameters with resolved paths, without rereading inputs."""
        content = json_bytes(data)
        with Path(path).open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _load_intrabeam_scattering(data: dict):
        """Validate IBS configurations and active named references."""
        from PASS.para.schema.ibs import load_intrabeam_scattering
        return load_intrabeam_scattering(data)

    @staticmethod
    def _intrabeam_scattering_engine_data(block):
        """Normalize IBS parameter keys while preserving configuration names."""
        return {
            "enabled": block.enabled,
            "configurations": {
                name: convert_keys_to_lower(config.model_dump(by_alias=True))
                for name, config in block.configurations.items()
            },
        }

    @staticmethod
    def _load_electron_cloud(data: dict):
        """Validate electron-cloud configurations and active named references."""
        from PASS.para.schema.electron_cloud import load_electron_cloud
        return load_electron_cloud(data)

    @staticmethod
    def _electron_cloud_engine_data(block):
        """Normalize parameter keys while preserving case-sensitive configuration names."""
        return {
            "enabled": block.enabled,
            "configurations": {
                name: convert_keys_to_lower(config.model_dump(by_alias=True))
                for name, config in block.configurations.items()
            },
        }

    @staticmethod
    def _load_space_charge(data: dict):
        """Parse the new top-level block, skipping configurations when disabled."""
        from PASS.para.schema.space_charge import SpaceChargeConfig, SpaceChargeResourceConfig

        keys = {str(key).casefold(): key for key in data}
        obsolete = [keys[name] for name in ("is space charge", "space-charge simulation parameters") if name in keys]
        if obsolete:
            raise ValueError("obsolete space-charge input key(s) "
                             f"{obsolete}; use the top-level 'Space charge' block")

        actual_key = keys.get("space charge")
        if actual_key is None:
            return SpaceChargeConfig(enabled=False), 0
        raw = data[actual_key]
        if not isinstance(raw, dict):
            raise TypeError("top-level 'Space charge' must be an object")

        raw_keys = {str(key).casefold(): key for key in raw}
        coverage_fields = {"coverage check", "coverage mode", "expected sc length (m)"}
        unknown = set(raw_keys) - {"enabled", "configurations", *coverage_fields}
        if unknown:
            names = [raw_keys[name] for name in sorted(unknown)]
            raise ValueError(f"unknown field(s) in 'Space charge': {names}")
        enabled_key = raw_keys.get("enabled")
        enabled = False if enabled_key is None else raw[enabled_key]
        if not isinstance(enabled, bool):
            raise TypeError("Space charge.Enabled must be boolean")

        configurations_key = raw_keys.get("configurations")
        raw_configurations = {} if configurations_key is None else raw[configurations_key]
        configuration_count = len(raw_configurations) if isinstance(raw_configurations, dict) else 0
        if not enabled:
            return SpaceChargeConfig(enabled=False), configuration_count

        # Enabled configurations are validated strictly, using the documented
        # fields. New-format key matching remains case-insensitive like the
        # rest of the engine. Disabled configuration contents are ignored.
        if not isinstance(raw_configurations, dict):
            raise TypeError("Space charge.Configurations must be an object")
        resource_keys = {}
        for field_name, field_info in SpaceChargeResourceConfig.model_fields.items():
            alias = field_info.alias or field_name
            resource_keys[field_name.casefold()] = alias
            resource_keys[str(alias).casefold()] = alias
        canonical_configurations = {}
        for name, sc_cfg in raw_configurations.items():
            if not isinstance(sc_cfg, dict):
                canonical_configurations[name] = sc_cfg
                continue
            canonical_configurations[name] = {resource_keys.get(str(k).casefold(), k): v for k, v in sc_cfg.items()}
        return SpaceChargeConfig.model_validate({
            "Enabled": enabled,
            "Configurations": canonical_configurations,
            **{
                SpaceChargeConfig.model_fields[field].alias: raw[raw_keys[key]]
                for key, field in (("coverage check", "coverage_check"), ("coverage mode", "coverage_mode"), ("expected sc length (m)", "expected_sc_length")) if key in raw_keys
            }
        }), configuration_count

    def get_log_path(self):
        return Path(self.output_dir_log) / f"{self.output_hms}.log"

    @staticmethod
    def _load_timing(raw_timing) -> dict:
        """Parse optional timing settings while keeping old inputs valid."""
        timing = {
            "mode": "command",
            "log_interval": 10,
            "warmup_turns": 1,
            "include_io": True,
        }
        if raw_timing is None:
            return timing
        if not isinstance(raw_timing, dict):
            logger.warning("Timing must be an object; using default timing settings.")
            return timing

        def get_value(*names, default):
            for name in names:
                if name in raw_timing:
                    return raw_timing[name]
            return default

        mode = str(get_value("mode", default=timing["mode"])).lower().strip()
        if mode not in {"off", "turn", "command", "synchronized-command"}:
            logger.warning("Timing mode must be one of off, turn, command, synchronized-command; "
                           "using 'command'.")
            mode = timing["mode"]
        timing["mode"] = mode

        interval = get_value("log interval", "log_interval", default=timing["log_interval"])
        if isinstance(interval, bool):
            interval = timing["log_interval"]
        try:
            interval = int(interval)
        except (TypeError, ValueError):
            interval = timing["log_interval"]
        if interval < 1:
            logger.warning("Timing log interval must be >= 1; using 10.")
            interval = timing["log_interval"]
        timing["log_interval"] = interval

        warmup = get_value("warmup turns", "warmup_turns", default=timing["warmup_turns"])
        if isinstance(warmup, bool):
            warmup = timing["warmup_turns"]
        try:
            warmup = int(warmup)
        except (TypeError, ValueError):
            warmup = timing["warmup_turns"]
        if warmup < 0:
            logger.warning("Timing warmup turns must be >= 0; using 0.")
            warmup = 0
        timing["warmup_turns"] = warmup

        include_io = get_value("include io", "include_io", default=timing["include_io"])
        if not isinstance(include_io, bool):
            logger.warning("Timing include io must be boolean; using true.")
            include_io = timing["include_io"]
        timing["include_io"] = include_io
        return timing

    def select_gpu_device(self) -> None:
        """Select the first configured CUDA device for this process.

        PASS currently executes one process on one GPU.  ``num_gpu`` and
        additional IDs are retained as input metadata, while the first ID is
        the device used for particle allocation and kernel launches.
        """
        if not self.use_gpu:
            return

        try:
            import cupy as cp
        except (ImportError, OSError) as exc:
            raise RuntimeError("The GPU backend was requested, but CuPy is unavailable. "
                               "Install PASS with the optional [cuda] extra.") from exc

        selected_gpu_id = int(self.gpu_id[0])
        device_count = cp.cuda.runtime.getDeviceCount()
        if selected_gpu_id < 0 or selected_gpu_id >= device_count:
            raise ValueError(f"Configured GPU device id {selected_gpu_id} is out of range "
                             f"for {device_count} visible CUDA device(s)")

        cp.cuda.Device(selected_gpu_id).use()

    def get_stat_path(self, beam_name, bunch_id):
        return Path(self.output_dir_stat / f"{beam_name}_bunch{bunch_id}_stat_{self.output_hms}")

    def get_dist_path(self, beam_name, bunch_id, turn):
        return Path(self.output_dir_dist / f"{beam_name}_bunch{bunch_id}_dist_turn{turn}_{self.output_hms}")

    def print(self):
        print_system_info()

        if self.use_gpu:
            print_cuda_system_info()
            print_cuda_device_info(self.gpu_id[0])

        set_simple_logging()

        logger.info("")
        logger.info(center_string(f" Configuration "))
        logger.info(f"Num Beam: {self.num_beam}")
        logger.info(f"Num Turn: {self.num_turn}")

        logger.info(f"Is Plot: {self.is_plot}")
        logger.info(f"Input Path: {self.input_path}")
        logger.info(f"Output ymd: {self.output_ymd}")
        logger.info(f"Output hms: {self.output_hms}")
        logger.info(f"Output Dir: {self.output_dir}")
        logger.info(f"Output Interval: {self.output_interval}")
        logger.info(f"Timing: {self.timing}")

        logger.info(f"Use CPU: {self.use_cpu}")
        logger.info(f"Use GPU: {self.use_gpu}")
        logger.info(f"Particle Precision: {self.particle_precision}")
        if self.use_gpu:

            logger.info(f"Num GPU: {self.num_gpu}")
            logger.info(f"GPU ID : {self.gpu_id}")

        set_normal_logging()


def print_system_info():

    set_simple_logging()

    logger.info("")
    logger.info(center_string(" System Information "))

    logger.info(f"Hostname              : {socket.gethostname()}")
    logger.info(f"OS                    : {platform.system()}")
    logger.info(f"Release               : {platform.release()}")
    logger.info(f"Version               : {platform.version()}")
    logger.info(f"Architecture          : {platform.machine()}")
    logger.info(f"Processor             : {platform.processor()}")
    logger.info(f"Python Version        : {platform.python_version()}")
    logger.info(f"Python Implementation : "
                f"{platform.python_implementation()}")
    logger.info(f"Python Compiler       : "
                f"{platform.python_compiler()}")
    logger.info(f"Executable            : {sys.executable}")
    logger.info(f"Current Directory     : {os.getcwd()}")
    logger.info(f"CPU Count             : {os.cpu_count()}")
    logger.info(f"Platform              : {platform.platform()}")
    logger.info(f"Node                  : {platform.node()}")
    logger.info(f"System Alias          : {platform.system_alias(*platform.uname()[:3])}")

    set_normal_logging()


def print_cuda_system_info():
    """
    NVML System Information
    """

    from cuda.core import Device, system
    from cuda.bindings import driver as cuda, runtime as cudart
    import platform

    set_simple_logging()

    logger.info("")
    logger.info(center_string(" Driver / NVML "))

    driver_major, driver_minor = system.get_user_mode_driver_version()
    logger.info(f"CUDA Driver Version : {driver_major}.{driver_minor}")

    err, runtime_ver = cudart.cudaRuntimeGetVersion()
    runtime_major = runtime_ver // 1000
    runtime_minor = (runtime_ver % 1000) // 10
    logger.info(f"CUDA Runtime Version: {runtime_major}.{runtime_minor}")

    num_devices = system.get_num_devices()
    logger.info(f"CUDA Device Count   : {num_devices}")

    devices = Device.get_all_devices()
    logger.info(f"Detected {len(devices)} CUDA Capable device(s)")

    for idx in range(num_devices):

        dev = system.Device(index=idx)

        logger.info("")
        logger.info(center_string(f" GPU[{idx}] : {dev.name} "))

        try:
            mem = dev.memory_info

            logger.info(f"Memory : "
                        f"{bytes_to_gb(mem.used)} GB / "
                        f"{bytes_to_gb(mem.total)} GB")
        except Exception as e:
            logger.warning(f"Memory query failed : {e}")

        try:
            temperature = dev.temperature.get_sensor()
            logger.info(f"Temperature : {temperature} °C")
        except Exception as e:
            logger.warning(f"Temperature query failed : {e}")

        try:
            logger.info(f"UUID : {dev.uuid}")
        except Exception:
            pass

        try:
            logger.info(f"P-State : {dev.performance_state}")
        except Exception:
            pass

    set_normal_logging()


def print_cuda_device_info(selected_gpu_id: int | None = None):
    """
    GPU Device Information
    """

    from cuda.core import Device
    import platform

    set_simple_logging()

    logger.info("")
    logger.info(center_string(f" CUDA Device Info "))

    devices = Device.get_all_devices()

    for idx, dev in enumerate(devices):
        props = dev.properties
        logger.info("")
        logger.info(center_string(f" GPU[{idx}] : {dev.name} "))

        # Basic Info
        logger.info(f"Compute Capability : {props.compute_capability_major}.{props.compute_capability_minor}")
        sm_cores = convert_sm_ver_to_cores(props.compute_capability_major, props.compute_capability_minor)
        total_cores = sm_cores * props.multiprocessor_count
        logger.info(f"SMs / CUDA cores per SM / Total cores : {props.multiprocessor_count} / {sm_cores} / {total_cores}")

        logger.info(f"Max threads per block : {props.max_threads_per_block}")
        logger.info(f"Max threads per multiprocessor : {props.max_threads_per_multiprocessor}")
        logger.info(f"Warp size : {props.warp_size}")

        # GPU Memory
        try:
            mem = dev.memory_info
            logger.info(f"Global memory total/free : "
                        f"{fmt_bytes(mem.total)}/{fmt_bytes(mem.total - mem.used)}")
        except Exception as e:
            logger.warning(f"Failed to get global memory: {e}")
        logger.info(f"Memory clock rate : {fmt_hz(props.memory_clock_rate)}")
        logger.info(f"Memory bus width : {props.global_memory_bus_width}-bit")
        logger.info(f"L2 cache size : {props.l2_cache_size/1024:.0f} KB")

        # Texture Info
        logger.info(f"Max 1D texture : {props.maximum_texture1d_width}")
        logger.info(f"Max 2D texture : {props.maximum_texture2d_width} x {props.maximum_texture2d_height}")
        logger.info(f"Max 3D texture : {props.maximum_texture3d_width} x {props.maximum_texture3d_height} x {props.maximum_texture3d_depth}")
        logger.info(f"Max 1D layered texture : {props.maximum_texture1d_layered_width} ({props.maximum_texture1d_layered_layers} layers)")
        logger.info(
            f"Max 2D layered texture : {props.maximum_texture2d_layered_width} x {props.maximum_texture2d_layered_height} ({props.maximum_texture2d_layered_layers} layers)"
        )

        # Memory and Registers
        logger.info(f"Total constant memory : {props.total_constant_memory} bytes")
        logger.info(f"Shared memory per block : {props.max_shared_memory_per_block} bytes")
        logger.info(f"Shared memory per multiprocessor : {props.max_shared_memory_per_multiprocessor} bytes")
        logger.info(f"Registers per block : {props.max_registers_per_block}")

        # Grid / Thread Limit
        logger.info(f"Max threads per block dim (x,y,z) : ({props.max_block_dim_x},{props.max_block_dim_y},{props.max_block_dim_z})")
        logger.info(f"Max grid size (x,y,z) : ({props.max_grid_dim_x},{props.max_grid_dim_y},{props.max_grid_dim_z})")
        logger.info(f"Max memory pitch : {props.max_pitch} bytes")
        logger.info(f"Texture alignment : {props.texture_alignment} bytes")

        # Functions and Modes
        logger.info(f"Concurrent copy and kernel execution : {yes_no(props.gpu_overlap)} with {props.async_engine_count} copy engine(s)")
        logger.info(f"Kernel execution timeout : {yes_no(props.kernel_exec_timeout)}")
        logger.info(f"Integrated GPU : {yes_no(props.integrated)}")
        logger.info(f"Can map host memory : {yes_no(props.can_map_host_memory)}")
        logger.info(f"ECC support : {'Enabled' if props.ecc_enabled else 'Disabled'}")
        if platform.system() == "Windows":
            logger.info(f"CUDA Device Driver Mode : {'TCC' if props.tcc_driver else 'WDDM'}")
        logger.info(f"Unified Addressing (UVA) : {yes_no(props.unified_addressing)}")
        logger.info(f"Managed memory : {yes_no(props.managed_memory)}")
        logger.info(f"Compute preemption supported : {yes_no(props.compute_preemption_supported)}")
        logger.info(f"Cooperative kernel launch support : {yes_no(props.cooperative_launch)}")
        logger.info(f"PCI Domain / Bus / Device : {props.pci_domain_id} / {props.pci_bus_id} / {props.pci_device_id}")

        # Calculation Mode
        compute_modes = {0: "Default", 1: "Exclusive", 2: "Prohibited", 3: "Exclusive Process"}
        logger.info(f"Compute Mode : {compute_modes.get(props.compute_mode,'Unknown')}")
        logger.info("")

    set_normal_logging()

    if selected_gpu_id is not None:
        import cupy as cp
        cp.cuda.Device(int(selected_gpu_id)).use()
        logger.info(f"Restored selected GPU device: {selected_gpu_id}")


def bytes_to_gb(nbytes):
    return round(nbytes / 1024**3, 2)


def fmt_bytes(size):
    return f"{size / (1024*1024):.0f} MB ({size} bytes)"


def fmt_hz(rate_khz):
    return f"{rate_khz*1e-3:.0f} MHz ({rate_khz*1e-6:.2f} GHz)"


def yes_no(val):
    return "Yes" if val else "No"


def convert_sm_ver_to_cores(major, minor):
    sm_to_cores = {
        (3, 0): 192,
        (3, 2): 192,
        (3, 5): 192,
        (3, 7): 192,
        (5, 0): 128,
        (5, 2): 128,
        (5, 3): 128,
        (6, 0): 64,
        (6, 1): 128,
        (6, 2): 128,
        (7, 0): 64,
        (7, 2): 64,
        (7, 5): 64,
        (8, 0): 64,
        (8, 6): 128,
        (8, 7): 128,
        (8, 9): 128,
        (9, 0): 128,
        (10, 0): 128,
        (10, 1): 128,
        (10, 3): 128,
        (11, 0): 128,
        (12, 0): 128,
        (12, 1): 128,
    }
    return sm_to_cores.get((major, minor), 0)
