from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


DEFAULT_EGAT_IMAGE = "erm42/yarp:egat"
DEFAULT_EGAT_COMMAND = "/opt/micromamba/bin/micromamba run -p /opt/egat-env python /opt/egat/egat_predict_reaction_csv.py"


def image_name_to_sif_name(image_name: str) -> str:
    return image_name.replace("/", "_").replace(":", "_") + ".sif"


DEFAULT_EGAT_SIF = image_name_to_sif_name(DEFAULT_EGAT_IMAGE)


@dataclass
class ResourceProfile:
    cpus: int = 8
    memory_mb: int = 8000
    disk_mb: int = 2048


@dataclass
class RetryConfig:
    infrastructure: int = 3
    chemistry: int = 1
    quarantine_after: int = 3


@dataclass
class OSGConfig:
    state_dir: str = ".yarp_osg"
    max_jobs: int = 10000
    submit_batch_size: int = 1000
    check_interval: int = 300
    container_directive: str = "container_image"
    osdf_namespace: str | None = None
    local_sif_dir: str | None = None
    egat_image: str = DEFAULT_EGAT_IMAGE
    egat_container: str | None = None
    egat_command: str | None = None
    egat_local_sif: str | None = None
    egat_resources: ResourceProfile = field(default_factory=ResourceProfile)
    retries: RetryConfig = field(default_factory=RetryConfig)

    @property
    def state_dir_name(self) -> str:
        return self.state_dir or ".yarp_osg"


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return {}
        value = value.get(key, {})
    return value


def _int_from(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    return int(value)


def _resource_from(data: Mapping[str, Any] | None, default: ResourceProfile) -> ResourceProfile:
    data = data or {}
    return ResourceProfile(
        cpus=_int_from(data.get("cpus"), default.cpus),
        memory_mb=_int_from(data.get("memory_mb"), default.memory_mb),
        disk_mb=_int_from(data.get("disk_mb"), default.disk_mb),
    )


def _default_local_sif_dir() -> str | None:
    repo_root = Path(__file__).resolve().parents[4]
    candidate = repo_root.parent / "containers" / "yarp_sifs_from_crc"
    return str(candidate) if candidate.is_dir() else None


def _default_osdf_namespace() -> str:
    return f"/ospool/ap40/data/{os.environ.get('USER', '$USER')}"


def _container_from_namespace(namespace: str | None, sif_name: str) -> str | None:
    if not namespace:
        return None
    namespace = namespace.strip()
    if namespace.startswith("osdf://"):
        base = namespace.rstrip("/")
    else:
        base = "osdf:///" + namespace.strip("/")
    return f"{base}/yarp-containers/v1/{sif_name}"


def load_osg_config(input_config: Mapping[str, Any] | None = None) -> OSGConfig:
    """Load helper config from raw YARP input config plus environment overrides."""
    input_config = input_config or {}
    osg_node = _nested(input_config, "initialize", "job_manager", "osg")
    if not isinstance(osg_node, Mapping):
        osg_node = {}

    containers = osg_node.get("containers", {}) or {}
    commands = osg_node.get("commands", {}) or {}
    resources = osg_node.get("resources", {}) or {}
    retries = osg_node.get("retries", {}) or {}

    default_resources = ResourceProfile()
    osdf_namespace = os.environ.get("YARP_OSG_OSDF_NAMESPACE") or osg_node.get("osdf_namespace") or _default_osdf_namespace()
    local_sif_dir = os.environ.get("YARP_OSG_LOCAL_SIF_DIR") or osg_node.get("local_sif_dir") or _default_local_sif_dir()
    egat_image = os.environ.get("YARP_OSG_EGAT_IMAGE") or containers.get("egat_image") or DEFAULT_EGAT_IMAGE
    egat_sif = image_name_to_sif_name(egat_image)
    egat_local_sif = str(Path(local_sif_dir) / egat_sif) if local_sif_dir else None
    egat_container = (
        os.environ.get("YARP_OSG_EGAT_CONTAINER")
        or containers.get("egat")
        or _container_from_namespace(osdf_namespace, egat_sif)
    )

    return OSGConfig(
        state_dir=str(osg_node.get("state_dir", ".yarp_osg")),
        max_jobs=_int_from(os.environ.get("YARP_OSG_MAX_JOBS", osg_node.get("max_jobs")), 10000),
        submit_batch_size=_int_from(osg_node.get("submit_batch_size"), 1000),
        check_interval=_int_from(osg_node.get("check_interval_seconds"), 300),
        container_directive=str(osg_node.get("container_directive", "container_image")),
        osdf_namespace=osdf_namespace,
        local_sif_dir=local_sif_dir,
        egat_image=egat_image,
        egat_container=egat_container,
        egat_command=os.environ.get("YARP_OSG_EGAT_COMMAND") or commands.get("egat") or DEFAULT_EGAT_COMMAND,
        egat_local_sif=egat_local_sif,
        egat_resources=_resource_from(resources.get("egat"), default_resources),
        retries=RetryConfig(
            infrastructure=_int_from(retries.get("infrastructure"), 3),
            chemistry=_int_from(retries.get("chemistry"), 1),
            quarantine_after=_int_from(retries.get("quarantine_after"), 3),
        ),
    )


def state_root(work_dir: Path, config: OSGConfig) -> Path:
    return work_dir / config.state_dir_name
