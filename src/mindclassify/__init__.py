"""mindclassify: a responsible benchmark for mental-health text classification. Research use only."""

__version__ = "0.1.0"

# The seven labels of the public corpus. "Suicidal" is never dropped.
LABELS = ("Normal", "Depression", "Suicidal", "Anxiety", "Bipolar", "Stress", "Personality disorder")
SAFETY_LABEL = "Suicidal"

DISCLAIMER = ("RESEARCH USE ONLY. This is not a diagnosis. A qualified person must review every case. "
              "If a person is in danger, contact local emergency services or a crisis line.")
