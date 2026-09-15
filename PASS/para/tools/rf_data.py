"""Physical-time RF tables and explicit synchronous-program construction."""
from pathlib import Path
import numpy as np
import tfs
from PASS.utils.program import LinearProgram
from PASS.utils.constants import const


def convert_rf_data(input_path: str, output_path: str) -> str:
    """Convert TIME, VOLTAGE, FREQUENCY, PHASE columns to physical-time TFS.

    Units are seconds, volts, Hz, radians. No implicit time-to-turn conversion.
    Phase is an unwrapped additive modulation; endpoint values are held.
    """
    import pandas as pd
    from PASS.para.schema.rf import RFComponent
    path=Path(input_path)
    table=tfs.read(path) if path.suffix.lower()=='.tfs' else pd.read_csv(path,sep=None,engine='python')
    table.columns=table.columns.str.lower()
    RFComponent(times=table.time.tolist(),voltage=table.voltage.tolist(),
                frequency=table.frequency.tolist(),phase=table.phase.tolist())
    result=tfs.TfsDataFrame({k.upper():table[k].to_numpy() for k in ('time','voltage','frequency','phase')})
    tfs.write(output_path,result,colwidth=25,headerswidth=25)
    return str(output_path)


def synchronous_rf_program(voltage, phase, harmonic, circumference, mass, kinetic_energy,
                           charge_per_nucleon=1., *, origin=0., time_origin=0.):
    """Build a prescribed waveform that hits requested phases at design passages.

    This is an input generator, not a runtime reset of bunch phase or energy.
    Between the design samples both frequency and additive phase are linear.
    origin is the first cavity passage; time_origin is the shared RF clock epoch.
    """
    voltage=np.asarray(voltage,dtype=float)
    phase=np.broadcast_to(np.asarray(phase,dtype=float),voltage.shape)
    times=[];frequencies=[];energy=float(kinetic_energy+mass);time=float(origin)
    for V,phi in zip(voltage,phase):
        beta=np.sqrt((energy-mass)*(energy+mass))/energy
        times.append(time);frequencies.append(harmonic*beta*const.c/circumference)
        energy+=charge_per_nucleon*V*np.sin(phi)
        if energy<=mass:raise ValueError('Design reference stops inside RF program')
        beta=np.sqrt((energy-mass)*(energy+mass))/energy
        time+=circumference/(beta*const.c)
    times=np.asarray(times);frequencies=np.asarray(frequencies)
    carrier=LinearProgram(frequencies,times,origin=time_origin)
    modulation=phase+2*np.pi*(harmonic*np.arange(len(times))-carrier.integral(0.,times))
    return tfs.TfsDataFrame(dict(TIME=times,VOLTAGE=voltage,FREQUENCY=frequencies,PHASE=modulation))
