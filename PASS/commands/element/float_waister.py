"""RFQ and ideal floating-waist maps with longitudinal companion kicks."""

from PASS.commands.command import Command
from PASS.commands.element.crab_cavity import _IPEquivalent
from PASS.para.schema.elements import FloatWaisterItem


@Command.register('floatwaister')
class FloatWaister(_IPEquivalent):

    def __init__(self, beam_id, sim, **command_kwargs):
        parameters = FloatWaisterItem.model_validate(command_kwargs)
        phases = parameters.phase_advance_x, parameters.phase_advance_y
        strengths = (parameters.gx, parameters.gy) if parameters.mode == 'rfq' else (parameters.strength_x, parameters.strength_y)
        self._initialize(beam_id, sim, parameters, parameters.mode, phases, strengths)
