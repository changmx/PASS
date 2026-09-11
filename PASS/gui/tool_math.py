"""Offline mathematical typesetting for Qt formula documents."""
from functools import lru_cache
from html import escape
from io import BytesIO
import re

from matplotlib import rc_context
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from PySide6.QtCore import QEvent, QUrl
from PySide6.QtGui import QImage, QPalette, QTextDocument
from PySide6.QtWidgets import QTextBrowser


@lru_cache(maxsize=384)
def equation_png(tex, color):
    buffer = BytesIO()
    display_tex = tex.strip().replace(r"\frac", r"\dfrac")
    with rc_context({"mathtext.fontset": "stix", "text.usetex": False, "savefig.transparent": True}):
        math_to_image(f"${display_tex}$", buffer, prop=FontProperties(size=18),
                      dpi=192, format="png", color=color)
    return buffer.getvalue()


class FormulaBrowser(QTextBrowser):
    """Render <eq>MathText</eq> blocks without a web engine or network assets."""
    def __init__(self, html, parent=None):
        super().__init__(parent)
        self.source_html = html
        self.formula_count = 0
        self.rendered_equations = []
        self.setOpenExternalLinks(True)
        self.document().setDocumentMargin(20)
        self.document().setDefaultStyleSheet("""
            body { font-size: 14px; }
            p { margin-top: 9px; margin-bottom: 12px; line-height: 150%; }
            h2 { font-size: 21px; margin-top: 22px; margin-bottom: 12px; }
            h3 { font-size: 17px; margin-top: 18px; }
            li { margin-bottom: 7px; }
            .equation { margin-top: 14px; margin-bottom: 18px; }
            a { text-decoration: underline; }
        """)

    def render(self):
        # Match the actual inherited application palette each time the window opens.
        color = self.palette().color(QPalette.Text).name()
        self.rendered_equations = []
        resources = []

        def replace(match):
            tex = match.group(1).strip()
            image = QImage.fromData(equation_png(tex, color))
            if image.isNull():
                raise ValueError(f"Cannot render formula: {tex}")
            url = QUrl(f"formula:{len(resources)}")
            # 192 dpi assets, 96 dpi logical layout: crisp at desktop scaling.
            width, height = image.width()/2, image.height()/2
            scale = min(1., max(240, self.viewport().width()-48)/width)
            resources.append((url, image))
            self.rendered_equations.append(tex)
            return (f'<p class="equation"><img src="{url.toString()}" '
                    f'width="{width*scale:.2f}" height="{height*scale:.2f}" '
                    f'alt="{escape(tex, quote=True)}" /></p>')

        html = re.sub(r"<eq>(.*?)</eq>", replace, self.source_html, flags=re.DOTALL)
        self.setHtml(html)
        for url, image in resources:
            self.document().addResource(QTextDocument.ImageResource, url, image)
        self.formula_count = len(resources)
        self.document().markContentsDirty(0, self.document().characterCount())

    def showEvent(self, event):
        super().showEvent(event)
        self.render()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.PaletteChange and self.isVisible() and hasattr(self, "source_html"):
            self.render()
