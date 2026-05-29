from gui.widgets.parameter_form import (
    ParameterForm
)


class InjectionPage(ParameterForm):

    def __init__(self, model):

        super().__init__(model)