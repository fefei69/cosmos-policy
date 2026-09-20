# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Important constants for Cosmos Policy training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.

Adapted from: https://github.com/user/openvla-oft/blob/main/experiments/robot/libero/run_libero_eval.py
"""

import os
import sys

# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 16,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 9,
}

ROBOCASA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 32,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 9,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 50,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
}

HANOI_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 63,
    "ACTION_DIM": 4,
    "PROPRIO_DIM": 4,
}

HANOI_JOINT_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 4,
    "PROPRIO_DIM": 7,
}

# Dense 10 Hz chunk of absolute reference poses (contract hanoi_dense_v5). The
# chunk length is the prepared dataset's horizon: 16 (decision 2) or 32 (the
# comparison run); HANOI_DENSE_HORIZON must match the dataset and checkpoint.
HANOI_DENSE_HORIZONS = (16, 32)
HANOI_DENSE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": int(os.environ.get("HANOI_DENSE_HORIZON", "16")),
    "ACTION_DIM": 4,
    "PROPRIO_DIM": 7,
}
if HANOI_DENSE_CONSTANTS["NUM_ACTIONS_CHUNK"] not in HANOI_DENSE_HORIZONS:
    raise ValueError(f"HANOI_DENSE_HORIZON must be one of {HANOI_DENSE_HORIZONS}")


# Function to detect robot platform from command line arguments
def detect_robot_platform():
    explicit = os.environ.get("COSMOS_POLICY_PLATFORM", "").upper()
    if explicit:
        if explicit not in {"LIBERO", "ROBOCASA", "ALOHA", "HANOI", "HANOI_JOINT", "HANOI_DENSE"}:
            raise ValueError(f"Unknown COSMOS_POLICY_PLATFORM: {explicit}")
        return explicit
    cmd_args = " ".join(sys.argv).lower()

    if "hanoi" in cmd_args:
        return "HANOI"
    elif "libero" in cmd_args:
        return "LIBERO"
    elif "robocasa" in cmd_args:
        return "ROBOCASA"
    elif "aloha" in cmd_args:
        return "ALOHA"
    else:
        # Default to LIBERO if unclear
        return "LIBERO"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ROBOCASA":
    constants = ROBOCASA_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "HANOI":
    constants = HANOI_CONSTANTS
elif ROBOT_PLATFORM == "HANOI_JOINT":
    constants = HANOI_JOINT_CONSTANTS
elif ROBOT_PLATFORM == "HANOI_DENSE":
    constants = HANOI_DENSE_CONSTANTS

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print("If needed, manually set the correct constants in `projects/cosmos/cosmos_policy/constants.py`!")
