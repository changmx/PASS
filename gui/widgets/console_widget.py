from PySide6.QtWidgets import (QPlainTextEdit)

from gui.core.logger import logger


class ConsoleWidget(QPlainTextEdit):

    def __init__(self):

        super().__init__()

        self.setReadOnly(True)

        # --------------------------------------
        # connect logger
        # --------------------------------------

        logger.message_logged.connect(self.on_message_logged)

    # ==========================================
    # slot
    # ==========================================

    def on_message_logged(
        self,
        message,
    ):

        self.appendPlainText(message)
