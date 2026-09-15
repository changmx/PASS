"""Group execution resources, separate from physics and checkpointable history."""
import numpy as np

from .convolution import ConvolutionGrid, PartitionedConvolution, ConvolutionState
from .time_convolution import TimeGrid, TimeConvolution, TimeConvolutionState
from .wake_state import WakeSources


class GroupExecution:
    """Persistent CPU/CUDA resources for one explicitly configured solver group."""
    def __init__(self, config, components, backend):
        self.config, self.components, self.backend = config, components, backend
        from .wake_solvers import FFTConvolutionSolver
        self.single_pass = FFTConvolutionSolver() if config.solver == "fft" else None
        self.convolution = None
        if config.solver == "partitioned_fft":
            self.convolution = PartitionedConvolution(components, ConvolutionGrid(**config.convolution_grid.model_dump()),
                config.memory_turns, backend=backend, method=config.partition or "dyadic",
                memory_time=config.memory_time, max_workspace_mb=config.max_workspace_mb or 1024)
        elif config.solver == "time_fft":
            self.convolution = TimeConvolution(components, TimeGrid(**config.time_grid.model_dump()),
                config.memory_time, backend=backend, method=config.partition or "dyadic",
                max_workspace_mb=config.max_workspace_mb or 1024)

    def preview(self, source, state, turn):
        cfg, components = self.config, self.components
        gpu = self.backend == "gpu"
        if gpu:
            import cupy as xp
            from .wake_state import DeviceSources as Sources
            from .wake_solvers import direct_gpu as direct
            from .wake_solvers import solve_wake_gpu as solve
        else:
            xp, Sources = np, WakeSources
            from .wake_solvers import DirectSliceSolver, solve_wake_cpu as solve
            direct = DirectSliceSolver().solve_cpu
        if cfg.source_shape == "point":
            if gpu:
                source = Sources(source.times, xp.zeros_like(source.widths), source.moments, source.betas, True,
                                 None if source.grid is None else (source.grid[0], 0.), source.increasing)
            else:
                source = Sources(source.times, xp.zeros_like(source.widths), source.moments, source.betas)
        update = None
        if self.convolution is not None:
            conv_state = state.convolution
            if isinstance(conv_state, dict):
                cls = TimeConvolutionState if cfg.solver == "time_fft" else ConvolutionState
                conv_state = cls.restore(conv_state, self.convolution)
            if conv_state is not None and state.last_turn != conv_state.start_turn+conv_state.count-1:
                raise ValueError("Convolution checkpoint turn does not match group state")
            values, update = self.convolution.preview(source, conv_state, turn=turn)
            candidate = state.fork()
            candidate.convolution, candidate.last_turn = update.state, turn
        elif cfg.boundary == "periodic":
            if state.last_turn is not None and turn <= state.last_turn:
                raise ValueError("Wake group turn did not advance")
            if gpu:
                from .wake_state import time_bounds_gpu,shifted_times_gpu
                if len(source.times):
                    lo,hi,width,_=time_bounds_gpu(self,source)
                    if hi-lo+width>cfg.period:raise ValueError('Periodic source domain exceeds one declared period')
                copies=[Sources(shifted_times_gpu(self,source.times,j*cfg.period),source.widths,source.moments,source.betas,source.point)
                        for j in range(-cfg.periodic_images,cfg.periodic_images+1)]
            else:
                if len(source.times) and float(xp.ptp(source.times)+xp.max(source.widths)) > cfg.period:
                    raise ValueError("Periodic source domain exceeds one declared period")
                copies = [Sources(source.times+j*cfg.period, source.widths, source.moments, source.betas)
                          for j in range(-cfg.periodic_images, cfg.periodic_images+1)]
            values = direct(components, copies, source.times, target_betas=source.betas)
            candidate = state.fork()
            candidate.last_turn = turn
        else:
            values, candidate = solve(components, source, state, turn=turn, solver=cfg.solver,
                memory_turns=0 if cfg.history == "none" else cfg.memory_turns, memory_time=cfg.memory_time,
                fft_plan=self.single_pass)
        diagnostics = {"name": cfg.name, "solver": cfg.solver, "history": cfg.history, "boundary": cfg.boundary,
            "retained_passages": len(candidate.history),
            "state_bytes": sum(v.nbytes for v in candidate.mode_amplitudes.values()),
            "fits": [getattr(c.model, "fit_diagnostics", None) for c in components]}
        if self.convolution is not None:
            diagnostics.update(self.convolution.diagnostics)
            diagnostics["state_bytes"] = candidate.convolution.nbytes
            diagnostics["processed_passages"] = candidate.convolution.count+1
            if cfg.solver == "time_fft":
                diagnostics["completed_time_blocks"] = update.blocks.start_turn+update.blocks.count+len(update.operations)
            else:
                diagnostics["history_horizon_turns"] = cfg.memory_turns
        return values, candidate, update, diagnostics
