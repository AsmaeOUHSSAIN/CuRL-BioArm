# Default workspace center for the myoArm reach task 
CENTER_X = -0.17
CENTER_Z = 1.40
DEFAULT_CENTER_Y = -0.25

# Square task parameter ranges (full / terminal difficulty).
SCALE_MIN = 0.02
SCALE_MAX = 0.08
ROTATION_MIN = -45.0
ROTATION_MAX = 45.0
Y_POS_MIN = -0.30
Y_POS_MAX = -0.12

# Default evaluation point.
EVAL_SCALE = 0.05
EVAL_ROTATION = 0.0
EVAL_Y_POS = -0.20

# Episode / training defaults.
MAX_EPISODE_STEPS = 500
TOLERANCE = 0.015

# Final evaluation suite.
FINAL_EVAL_LEVELS = {
    "large_norot_center": {"scale": 0.08, "rotation": 0.0,  "y_pos": -0.20, "tolerance": 0.01},
    "medium_midrot_mid":  {"scale": 0.05, "rotation": 20.0, "y_pos": -0.16, "tolerance": 0.01},
    "small_highrot_far":  {"scale": 0.02, "rotation": 40.0, "y_pos": -0.28, "tolerance": 0.01},
}
