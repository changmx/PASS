import os

project = 'PASS'
author = 'PASS Team'

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
]

# 主题（推荐科研风）
html_theme = "sphinx_rtd_theme"

# ⭐ logo（注意 assets 在 docs/assets）
html_logo = "../assets/logo_blue.png"

html_static_path = ['_static']

html_css_files = [
    'custom.css',
]

templates_path = ["_templates"]

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}

# ⭐ 让路径正确（重要）
html_baseurl = ""

copyright = "2025-2026 Institute of Modern Physics, Chinese Academy of Sciences"