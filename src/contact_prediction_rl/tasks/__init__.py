# src/contact_prediction_rl/tasks/__init__.py

"""Task registrations for the generated project."""

from isaaclab_tasks.utils import import_packages

import_packages(__name__, ["utils", ".mdp"])
