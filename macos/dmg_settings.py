"""Disk image layout for dmgbuild. No license dialog, no Finder scripting."""
import os

application = defines.get("app")  # noqa: F821
leeme = defines.get("leeme")  # noqa: F821
appname = os.path.basename(application)

format = "UDZO"
files = [application]
if leeme:
    files.append(leeme)
symlinks = {"Applications": "/Applications"}
icon = os.path.join(application, "Contents", "Resources", "AppIcon.icns")
icon_locations = {
    appname: (160, 170),
    "Applications": (480, 170),
    "LEEME.txt": (320, 340),
}
background = "builtin-arrow"
show_status_bar = False
show_tab_view = False
show_toolbar = False
show_pathbar = False
show_sidebar = False
window_rect = ((120, 80), (680, 460))
default_view = "icon-view"
icon_size = 110
text_size = 14
