"""Fetch immutable public files; reject corrupt caches and incomplete downloads."""

import hashlib
import json
import os
from pathlib import Path
import urllib.request
from .config import ROOT


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(data_dir, groups=("robot", "policy", "vlm"), offline=False):
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    for group in groups:
        for entry in lock[group]["files"]:
            path = Path(data_dir) / group / entry["path"]
            if path.is_file() and sha256(path) == entry["sha256"]:
                continue
            if offline:
                raise RuntimeError(f"Missing or corrupt locked file: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + f".{os.getpid()}.partial")
            print(f"Downloading {group}/{entry['path']}", flush=True)
            try:
                request = urllib.request.Request(
                    entry["url"], headers={"User-Agent": "so101-lab/0.1"}
                )
                with (
                    urllib.request.urlopen(request, timeout=120) as src,
                    temp.open("wb") as dst,
                ):
                    while block := src.read(1024 * 1024):
                        dst.write(block)
                if (
                    temp.stat().st_size != entry["size"]
                    or sha256(temp) != entry["sha256"]
                ):
                    raise RuntimeError(f"Checksum mismatch: {entry['url']}")
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
    return lock


def robot_urdf(data_dir):
    return Path(data_dir).resolve() / "robot/Simulation/SO101/so101_new_calib.urdf"


def prepare_import_urdf(data_dir, destination):
    """Drop the empty TCP marker from the physics copy, preserving all six joints.

    The pinned URDF remains untouched and defines FK. The leaf TCP has no geometry
    and zero inertia; importing it as a rigid body is unnecessary and unstable.
    """
    import xml.etree.ElementTree as ET

    source = robot_urdf(data_dir)
    tree = ET.parse(source)
    root = tree.getroot()
    marker = root.find("link[@name='gripper_frame_link']")
    joint = root.find("joint[@name='gripper_frame_joint']")
    if (
        marker is None
        or joint is None
        or joint.get("type") != "fixed"
        or marker.find("visual") is not None
        or marker.find("collision") is not None
        or any(
            j.find("parent").get("link") == "gripper_frame_link"
            for j in root.findall("joint")
        )
    ):
        raise ValueError("Unexpected TCP marker structure in the locked URDF.")
    root.remove(marker)
    root.remove(joint)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for mesh in root.iter("mesh"):
        mesh.set(
            "filename",
            os.path.relpath(source.parent / mesh.get("filename"), destination.parent),
        )
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination
