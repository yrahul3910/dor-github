#!/usr/bin/env python
import os
import sys

if __name__ == "__main__":
    # On startup, delete cache.
    # Uncomment for latest results. For paper results, leave commented.
    #if os.path.exists('dor-cache.sqlite'):
    #    os.remove('dor-cache.sqlite')

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gettingstarted.settings")

    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)
