"""Physical RF waveform inputs. Tables use seconds, volts, Hz and radians."""
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class ReferenceClock(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='forbid', allow_inf_nan=False)
    origin: float = Field(default=0., alias='Time origin (s)')
    frequency: float | list[float] = Field(alias='Revolution frequency (Hz)')
    times: list[float] | None = Field(default=None, alias='Time (s)')

    @model_validator(mode='after')
    def validate_program(self):
        from PASS.utils.program import LinearProgram
        p = LinearProgram(self.frequency, self.times, origin=self.origin)
        if (p.values <= 0).any():
            raise ValueError('Reference frequency must be positive')
        return self


class RFComponent(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='forbid', allow_inf_nan=False)
    voltage: float | list[float] = Field(default=0., alias='Voltage (V)')
    frequency: float | list[float] | None = Field(default=None, alias='Frequency (Hz)')
    harmonic: StrictInt | None = Field(default=None, ge=1, alias='Harmonic')
    phase: float | list[float] = Field(default=0., alias='Phase (rad)')
    times: list[float] | None = Field(default=None, alias='Time (s)')
    program_file: str | None = Field(default=None, alias='Program file')

    @model_validator(mode='after')
    def validate_program(self):
        if self.program_file is not None:
            if self.frequency is not None or self.times is not None or self.voltage != 0. or self.phase != 0.:
                raise ValueError('Program file cannot be combined with inline waveform data')
            return self
        if (self.frequency is None) == (self.harmonic is None):
            raise ValueError('Specify exactly one of Frequency (Hz) and Harmonic')
        from PASS.utils.program import LinearProgram
        for value in (self.voltage, self.phase):
            LinearProgram(value, self.times)
        if self.frequency is not None:
            p = LinearProgram(self.frequency, self.times)
            if (p.values <= 0).any():
                raise ValueError('RF frequency must be positive')
        return self
