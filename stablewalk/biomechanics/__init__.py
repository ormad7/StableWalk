"""
Biomechanics adapter layer — skeleton motion ↔ OpenSim-ready formats.

Primary entry points::

    from stablewalk.biomechanics import (
        BiomechanicsService,
        export_to_opensim_format,
        load_opensim_motion_data,
        map_skeleton_to_opensim_joints,
        calculate_joint_angles,
        calculate_degrees_of_freedom,
    )

CamelCase aliases (``exportToOpenSimFormat``, etc.) are provided for API
documentation parity with OpenSim tooling.
"""

from stablewalk.biomechanics.service import (
    BiomechanicsService,
    calculateJointAngles,
    calculateDegreesOfFreedom,
    calculate_degrees_of_freedom,
    calculate_joint_angles,
    exportToOpenSimFormat,
    export_to_opensim_format,
    loadOpenSimMotionData,
    load_opensim_motion_data,
    mapSkeletonToOpenSimJoints,
    map_skeleton_to_opensim_joints,
)
from stablewalk.biomechanics.types import (
    OpenSimExportBundle,
    OpenSimJointMapping,
    OpenSimMarkerTable,
    OpenSimMotionData,
    OpenSimMotionTable,
)

__all__ = [
    "BiomechanicsService",
    "OpenSimExportBundle",
    "OpenSimJointMapping",
    "OpenSimMarkerTable",
    "OpenSimMotionData",
    "OpenSimMotionTable",
    "export_to_opensim_format",
    "exportToOpenSimFormat",
    "load_opensim_motion_data",
    "loadOpenSimMotionData",
    "map_skeleton_to_opensim_joints",
    "mapSkeletonToOpenSimJoints",
    "calculate_joint_angles",
    "calculateJointAngles",
    "calculate_degrees_of_freedom",
    "calculateDegreesOfFreedom",
]
