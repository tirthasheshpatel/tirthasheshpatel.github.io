#!/usr/bin/env python
# -*- coding: utf-8 -*- #

# See https://docs.getpelican.com/en/latest/content.html for a long list and documentation of options!
# Also see https://docs.getpelican.com/en/latest/settings.html for a full list

AUTHOR = 'Tirth Patel'
SITENAME = 'Tirth Patel'
SITESUBTITLE = 'Blogs and More!'
SITEURL = ''

PATH = 'content'

TIMEZONE = 'Asia/Kolkata'

DEFAULT_LANG = 'en'

THEME = './notmyidea-modified'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
#LINKS = (('Pelican', 'https://getpelican.com/'),
#         ('Python.org', 'https://www.python.org/'),
#         ('Jinja2', 'https://palletsprojects.com/p/jinja/'),
#         ('GitHub', 'https://www.github.com/tirthasheshpatel'),)

# Social widget
#SOCIAL = (('My Home', '#'),
#          ('Floor gang', 'https://tirthasheshpatel.github.io'),)
#SOCIAL_WIDGET_NAME = "follow"

# Uncomment to generate a "Fork on GitHub" widget at the top of website
# Disabled because it's wierd in mobile web
#GITHUB_URL = 'https://www.github.com/tirthasheshpatel/tirthasheshpatel.github.io'

# WARNING: support for pagination removed. Don't set to true!
DEFAULT_PAGINATION = False

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

PLUGINS = ["pelican_alias",
           "render_math"]

# display pages on menu?
#DISPLAY_PAGES_ON_MENU = False

# display categories on menu?
DISPLAY_CATEGORIES_ON_MENU = False

# Delete the output directory, and all of its contents, before generating
# new files. This can be useful in preventing older, unnecessary files
# from persisting in your output. However, this is a destructive setting
# and should be handled with extreme care.
#DELETE_OUTPUT_DIRECTORY = True

# Static files are files other than articles and pages that are copied
# to the output folder as-is, without processing. You can control which
# static files are copied over with the STATIC_PATHS setting of the
# project’s pelicanconf.py file. Pelican’s default configuration includes
# the images directory for this, but others must be added manually. In
# addition, static files that are explicitly linked to are included.
# Starting with Pelican 3.5, static files can safely share a source
# directory with page source files, without exposing the page sources in
# the generated site. Any such directory must be added to both
# STATIC_PATHS and PAGE_PATHS
STATIC_PATHS = ['favicon.ico', 'pdfs', 'images']

# A list of glob patterns. Files and directories matching any of these
# patterns will be ignored by the processor. For example, the default
# ['.#*'] will ignore emacs lock files, and ['__pycache__'] would ignore
# Python 3’s bytecode caches.
IGNORE_FILES = ['*.tirth.excluded', '*.swp', 'draft_*.md']

# Generate URLs
ARTICLE_URL = 'blogs/{slug}/'
ARTICLE_SAVE_AS = 'blogs/{slug}/index.html'
PAGE_URL = '{slug}/'
PAGE_SAVE_AS = '{slug}/index.html'
DRAFT_URL = 'drafts/{slug}/'
DRAFT_SAVE_AS = 'drafts/{slug}/index.html'
AUTHOR_URL = 'author/{slug}/'
AUTHOR_SAVE_AS = 'author/{slug}/index.html'
CATEGORY_URL = 'category/{slug}/'
CATEGORY_SAVE_AS = 'category/{slug}/index.html'
TAG_URL = 'tag/{slug}/'
TAG_SAVE_AS = 'tag/{slug}/index.html'

# Year Archive
YEAR_ARCHIVE_SAVE_AS = 'blogs/{date:%Y}/index.html'
MONTH_ARCHIVE_SAVE_AS = 'blogs/{date:%Y}/{date:%b}/index.html'

# Markdown dictionary!
#MARKDOWN = {
#    'extension_configs': {
#        'markdown.extensions.codehilite': {'css_class': 'highlight'},
#        'markdown.extensions.extra': {},
#        'markdown.extensions.meta': {},
#        'markdown.extensions.admonition': {},
#    },
#    'output_format': 'html5',
#}

# Typogrify
TYPOGRIFY = True
