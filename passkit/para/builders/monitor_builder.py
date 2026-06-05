class MonitorBuilder:

    @staticmethod
    def build(model):

        result = {}

        if model.enable_stat_monitor:

            result["StatMonitor_start"] = {
                "S (m)": 0,
                "Command": "StatMonitor",
            }

        return result
