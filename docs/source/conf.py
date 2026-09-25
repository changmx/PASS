from functools import lru_cache
from gettext import translation
from pathlib import Path
import re
import tomllib

import sphinx
import sphinx_rtd_theme
from sphinx import addnodes
from sphinx.environment.adapters.toctree import TocTree
from sphinx.search.en import SearchEnglish

project = "PASS"
author = "PASS Team"
release = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
version = release
copyright = "2025-2026 Institute of Modern Physics, Chinese Academy of Sciences"

extensions = ["sphinx.ext.mathjax", "sphinx.ext.autodoc", "sphinx.ext.viewcode"]
html_theme = "sphinx_rtd_theme"
html_logo = "../assets/logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
templates_path = ["_templates"]
html_theme_options = {"collapse_navigation": True, "navigation_depth": 3}
html_show_sourcelink = False
html_search_language = "pass_bilingual"


class _BilingualSearch(SearchEnglish):
    """Index English words and overlapping Chinese character pairs."""

    lang = "pass_bilingual"
    # Sphinx uses this name to select the inherited JavaScript stemmer.
    language_name = "English"
    js_splitter_code = r"""
function splitQuery(query) {
    const words = query.match(/[A-Za-z0-9_]+|[\u3400-\u9fff]+/g) || [];
    return words.flatMap(word => {
        if (!/[\u3400-\u9fff]/.test(word) || word.length < 2) return [word];
        return Array.from({length: word.length - 1}, (_, i) => word.slice(i, i + 2));
    });
}
"""

    def split(self, text):
        words = []
        for word in re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]+", text):
            if re.search(r"[\u3400-\u9fff]", word) and len(word) > 1:
                words.extend(word[i:i + 2] for i in range(len(word) - 1))
            else:
                words.append(word)
        return words


@lru_cache(maxsize=2)
def _page_translation(language):
    translator = translation("sphinx", Path(sphinx_rtd_theme.__file__).parent / "locale", languages=[language], fallback=True)
    translator.add_fallback(translation("sphinx", Path(sphinx.__file__).parent / "locale", languages=[language], fallback=True))
    return translator


def _set_page_language(app, pagename, templatename, context, doctree):
    language = "zh_CN" if pagename.startswith("zh/") else "en"
    context["language"] = language
    translator = _page_translation(language)
    context["_"] = translator.gettext
    context["gettext"] = translator.gettext
    context["ngettext"] = translator.ngettext
    if pagename.startswith(("en/", "zh/")):
        prefix, suffix = pagename.split("/", 1)
        other = "en" if prefix == "zh" else "zh"
        counterpart = f"{other}/{suffix}"
        context["language_target"] = counterpart if counterpart in app.env.found_docs else f"{other}/index"
        context["master_doc"] = f"{prefix}/index"
        context["docstitle"] = f"PASS {release} 使用手册" if prefix == "zh" else f"PASS {release} User Manual"
        navigation = []
        adapter = TocTree(app.env)
        for tree in app.env.get_doctree(f"{prefix}/index").findall(addnodes.toctree):
            resolved = adapter.resolve(pagename, app.builder, tree, maxdepth=3, collapse=True, includehidden=True)
            if resolved is not None:
                navigation.append(app.builder.render_partial(resolved)["fragment"])
        context["language_navigation"] = "\n".join(navigation)


def setup(app):
    app.add_search_language(_BilingualSearch)
    app.connect("html-page-context", _set_page_language)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
