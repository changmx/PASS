from datetime import datetime

from PySide6.QtCore import (
    QObject,
    Signal,
)


class Logger(QObject):

    # ==========================================
    # Qt signal
    # ==========================================

    message_logged = Signal(str)

    # ==========================================
    # init
    # ==========================================

    def __init__(self):

        super().__init__()

        self.enable_gui = False

    # ==========================================
    # timestamp
    # ==========================================

    def timestamp(self):

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ==========================================
    # format
    # ==========================================

    def format_message(
        self,
        level,
        text,
    ):

        return (f"[{self.timestamp()}]"
                f"[{level}] "
                f"{text}")

    # ==========================================
    # emit
    # ==========================================

    def emit_message(
        self,
        level,
        text,
    ):

        message = self.format_message(
            level,
            text,
        )

        # --------------------------------------
        # terminal
        # --------------------------------------

        print(message)

        # --------------------------------------
        # gui
        # --------------------------------------

        if self.enable_gui:

            self.message_logged.emit(message)

    # ==========================================
    # levels
    # ==========================================

    def debug(self, text):

        self.emit_message(
            "DEBUG",
            text,
        )

    def info(self, text):

        self.emit_message(
            "INFO",
            text,
        )

    def warning(self, text):

        self.emit_message(
            "WARNING",
            text,
        )

    def error(self, text):

        self.emit_message(
            "ERROR",
            text,
        )


# =================================================
# global singleton
# =================================================

logger = Logger()
