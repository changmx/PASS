from PASS.utils.logger import set_simple_logging, set_normal_logging, center_string
from PASS.utils.helper import convert_keys_to_lower

from dataclasses import dataclass, field
from typing import Literal
from datetime import datetime
from pathlib import Path
import json
import shutil
import os
import sys
import socket
import platform
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:

    num_beam: int = 0
    beam_name: list[str] = field(default_factory=list)
    num_bunch: list[int] = field(default_factory=list)
    num_turn: int = 0
    num_collision: int = 0
    use_cpu: bool = False
    use_gpu: bool = True
    num_gpu: int = 0
    gpu_id: list[int] = field(default_factory=list)
    is_plot: bool = False
    input_path: list[str] = field(default_factory=list)
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
    output_dir_plot: str = ""
    output_dir_particle: str = ""
    output_dir_slowExt_particle: str = ""

    def load_input(self, beam0_path: str, beam1_path: str | None = None) -> None:
        path0 = Path(beam0_path)
        if not path0.exists():
            raise FileNotFoundError(f"Input beam0 file not found: {path0}")
        with open(path0, 'r', encoding='utf-8') as f:
            data0 = json.load(f)
            data0 = convert_keys_to_lower(data0)

        if beam1_path is not None:
            path1 = Path(beam1_path)
            if not path1.exists():
                raise FileNotFoundError(f"Input beam1 file not found: {path1}")
            with open(path1, 'r', encoding='utf-8') as f:
                data1 = json.load(f)
                data1 = convert_keys_to_lower(data1)

        if beam1_path is None:
            self.num_beam = 1
        else:
            self.num_beam = 2

        if self.num_beam == 1:
            self.beam_name.append(data0.get("beam name"))
            self.input_path.append(beam0_path)
            self.num_bunch.append(len(data0["sequence"]["injection"]) - 2)
        else:
            self.beam_name.append(data0.get("beam name"))
            self.beam_name.append(data1.get("beam name"))
            self.input_path.append(beam0_path)
            self.input_path.append(beam1_path)
            self.num_bunch.append(len(data0["sequence"]["injection"]) - 2)
            self.num_bunch.append(len(data1["sequence"]["injection"]) - 2)

        self.num_turn = data0.get("number of turns", 0)
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

        if self.use_gpu:
            self.num_gpu = data0.get("number of gpu devices")
            self.gpu_id = data0.get("device id")

        self.is_plot = data0.get("is plot figure")

        output_base = data0.get("output directory")
        now = datetime.now()
        self.output_ymd = f"{now.year}_{now.month:02d}{now.day:02d}"

        hour_min = f"{now.hour:02d}{now.minute:02d}"
        second = f"{now.second:02d}"
        base_hms = f"{hour_min}_{second}"
        self.output_hms = base_hms
        self.output_dir = str(Path(output_base) / self.output_ymd / self.output_hms)

        if Path(self.output_dir).exists():
            micro = now.microsecond // 1000
            self.output_hms = f"{hour_min}_{second}.{micro:03d}"
            self.output_dir = str(Path(output_base) / self.output_ymd / self.output_hms)

        max_attempts = 10
        attempt = 0
        while Path(self.output_dir).exists() and attempt < max_attempts:
            time.sleep(1)
            now = datetime.now()
            hour_min = f"{now.hour:02d}{now.minute:02d}"
            second = f"{now.second:02d}"
            micro = now.microsecond // 1000
            self.output_hms = f"{hour_min}_{second}.{micro:03d}"
            self.output_dir = str(Path(output_base) / self.output_ymd / self.output_hms)
            attempt += 1

        self.output_dir_log = str(Path(output_base) / self.output_ymd / self.output_hms)
        self.output_dir_stat = str(Path(output_base) / self.output_ymd / self.output_hms)
        self.output_dir_para = str(Path(output_base) / self.output_ymd / self.output_hms)
        self.output_dir_dist = str(Path(output_base) / self.output_ymd / self.output_hms / "distribution")
        self.output_dir_tuneSpread = str(Path(output_base) / self.output_ymd / self.output_hms / "tuneSpread")
        self.output_dir_chargeDensity = str(Path(output_base) / self.output_ymd / self.output_hms / "chargeDensity")
        self.output_dir_plot = str(Path(output_base) / self.output_ymd / self.output_hms / "plot")
        self.output_dir_particle = str(Path(output_base) / self.output_ymd / self.output_hms / "particle")
        self.output_dir_slowExt_particle = str(Path(output_base) / self.output_ymd / self.output_hms / "slowExt_particle")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir_log).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir_stat).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir_para).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir_dist).mkdir(parents=True, exist_ok=True)

        shutil.copy(beam0_path, Path(self.output_dir_para) / f"beam0_{self.output_hms}.json")
        if beam1_path is not None:
            shutil.copy(beam1_path, Path(self.output_dir_para) / f"beam1_{self.output_hms}.json")

    def get_log_path(self):
        return Path(self.output_dir_log) / f"{self.output_hms}.log"

    def get_stat_path(self, beam_name, bunch_id):
        return Path(self.output_dir_stat / f"{beam_name}_bunch{bunch_id}_stat_{self.output_hms}")

    def get_dist_path(self, beam_name, bunch_id, turn):
        return Path(self.output_dir_dist / f"{beam_name}_bunch{bunch_id}_dist_turn{turn}_{self.output_hms}")

    def print(self):
        print_system_info()

        if self.use_gpu:
            print_cuda_system_info()
            print_cuda_device_info()

        set_simple_logging()

        logger.info("")
        logger.info(center_string(f" Configuration "))
        logger.info(f"Num Beam: {self.num_beam}")
        logger.info(f"Num Turn: {self.num_turn}")
        # logger.info(f"Num collision : {self.num_collision}")

        logger.info(f"Is Plot: {self.is_plot}")
        logger.info(f"Input Path: {self.input_path}")
        logger.info(f"Output ymd: {self.output_ymd}")
        logger.info(f"Output hms: {self.output_hms}")
        logger.info(f"Output Dir: {self.output_dir}")
        logger.info(f"Output Interval: {self.output_interval}")

        logger.info(f"Use CPU: {self.use_cpu}")
        logger.info(f"Use GPU: {self.use_gpu}")
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
            temp = dev.temperature.get_sensor()
            logger.info(f"Temperature : {temp} °C")
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


def print_cuda_device_info():
    """
    GPU Device Information
    """

    from cuda.core import Device, system
    from cuda.bindings import driver as cuda, runtime as cudart
    import platform

    set_simple_logging()

    logger.info("")
    logger.info(center_string(f" CUDA Device Info "))

    devices = Device.get_all_devices()

    for idx, dev in enumerate(devices):
        props = dev.properties
        dev.set_current()
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
            err, free_mem, total_mem = cuda.cuMemGetInfo()
            logger.info(f"Global memory total/free : {fmt_bytes(total_mem)}/{fmt_bytes(free_mem)}")
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

    set_normal_logging()


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
