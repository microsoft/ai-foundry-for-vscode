from __future__ import annotations

from copy import deepcopy
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen


REPO_ROOT = Path(os.environ["REPO_ROOT"]) if "REPO_ROOT" in os.environ else Path(__file__).resolve().parents[2]
PLACEHOLDER = "{{SafeProjectName}}"
UPSTREAM_REPO_API = "https://api.github.com/repos/microsoft/agent-framework/contents"
ALLOWED_SUFFIXES = {".cs", ".csproj"}


@dataclass(frozen=True)
class SampleConfig:
    upstream_name: str
    target_dir: Path

    @property
    def upstream_path(self) -> str:
        return f"dotnet/samples/05-end-to-end/HostedAgents/{self.upstream_name}"


SAMPLES = (
    SampleConfig(
        upstream_name="FoundrySingleAgent",
        target_dir=REPO_ROOT / "samples" / "hosted-agent" / "dotnet" / "agent",
    ),
    SampleConfig(
        upstream_name="FoundryMultiAgent",
        target_dir=REPO_ROOT / "samples" / "hosted-agent" / "dotnet" / "workflow",
    ),
)


GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def _auth_headers() -> dict[str, str]:
    headers: dict[str, str] = {"User-Agent": "microsoft-foundry-for-vscode-sync"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def fetch_json(url: str) -> list[dict[str, object]]:
    headers = _auth_headers()
    headers["Accept"] = "application/vnd.github+json"
    request = Request(url, headers=headers)
    with urlopen(request) as response:
        return json.load(response)


def fetch_text(url: str) -> str:
    request = Request(url, headers=_auth_headers())
    with urlopen(request) as response:
        return response.read().decode("utf-8")


def normalize_content(content: str, upstream_name: str) -> str:
    return content.replace(upstream_name, PLACEHOLDER)


def normalize_relative_path(relative_path: Path, upstream_name: str) -> Path:
    normalized_parts = [part.replace(upstream_name, PLACEHOLDER) for part in relative_path.parts]
    return Path(*normalized_parts)


def list_relevant_files(upstream_path: str) -> list[dict[str, object]]:
    items = fetch_json(f"{UPSTREAM_REPO_API}/{upstream_path}?ref=main")
    relevant_files: list[dict[str, object]] = []

    for item in items:
        item_type = item.get("type")
        item_path = str(item["path"])

        if item_type == "dir":
            relevant_files.extend(list_relevant_files(item_path))
            continue

        if item_type != "file":
            continue

        if Path(str(item["name"])).suffix not in ALLOWED_SUFFIXES:
            continue

        relevant_files.append(item)

    return relevant_files


def replace_element_contents(target: ET.Element, source: ET.Element) -> None:
    target.attrib.clear()
    target.attrib.update(source.attrib)
    target.text = source.text
    target.tail = source.tail

    for child in list(target):
        target.remove(child)

    for child in source:
        target.append(deepcopy(child))


def element_attribute_key(element: ET.Element) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(element.attrib.items()))


def find_or_create_matching_group(root: ET.Element, source_group: ET.Element, tag: str) -> ET.Element:
    group_key = element_attribute_key(source_group)

    for existing_group in root.findall(tag):
        if element_attribute_key(existing_group) == group_key:
            return existing_group

    new_group = ET.Element(tag, source_group.attrib)
    insert_at = sum(1 for child in root if child.tag == tag)
    root.insert(insert_at, new_group)
    return new_group


def merge_csproj_content(upstream_content: str, local_path: Path) -> str:
    """Merge upstream .csproj changes into the local file, preserving local additions.

    Only PropertyGroup properties and ItemGroup PackageReferences from upstream
    are compared and updated.  Local-only properties and packages are kept as-is.
    If the local file does not exist or cannot be parsed, the full upstream
    content is returned unchanged.
    """
    if not local_path.exists():
        return upstream_content

    local_content = local_path.read_text(encoding="utf-8")

    try:
        upstream_root = ET.fromstring(upstream_content)
        local_root = ET.fromstring(local_content)
    except ET.ParseError:
        return upstream_content

    # --- Merge PropertyGroup ---
    for upstream_pg in upstream_root.findall("PropertyGroup"):
        local_pg = find_or_create_matching_group(local_root, upstream_pg, "PropertyGroup")
        local_props = {prop.tag: prop for prop in local_pg}

        for upstream_prop in upstream_pg:
            if upstream_prop.tag in local_props:
                replace_element_contents(local_props[upstream_prop.tag], upstream_prop)
            else:
                local_pg.append(deepcopy(upstream_prop))

    # --- Merge PackageReference items ---
    upstream_packages: dict[str, ET.Element] = {}
    for ig in upstream_root.findall("ItemGroup"):
        for pr in ig.findall("PackageReference"):
            include = pr.get("Include")
            if include:
                upstream_packages[include] = deepcopy(pr)

    if upstream_packages:
        # Find the first ItemGroup that already contains PackageReferences
        local_pkg_ig: ET.Element | None = None
        for ig in local_root.findall("ItemGroup"):
            if ig.findall("PackageReference"):
                local_pkg_ig = ig
                break

        if local_pkg_ig is None:
            local_pkg_ig = ET.SubElement(local_root, "ItemGroup")

        local_pkg_map: dict[str, ET.Element] = {}
        for ig in local_root.findall("ItemGroup"):
            for pr in ig.findall("PackageReference"):
                include = pr.get("Include")
                if include:
                    local_pkg_map[include] = pr

        for include, upstream_package in upstream_packages.items():
            if include in local_pkg_map:
                replace_element_contents(local_pkg_map[include], upstream_package)
            else:
                local_pkg_ig.append(deepcopy(upstream_package))

    ET.indent(local_root, space="  ")
    return ET.tostring(local_root, encoding="unicode", xml_declaration=False) + "\n"


def sync_sample(config: SampleConfig) -> list[Path]:
    items = list_relevant_files(config.upstream_path)
    expected_paths: list[Path] = []

    for item in items:
        download_url = str(item["download_url"])
        relative_upstream_path = Path(str(item["path"])).relative_to(config.upstream_path)
        target_relative_path = normalize_relative_path(relative_upstream_path, config.upstream_name)
        target_path = config.target_dir / target_relative_path
        normalized_content = normalize_content(fetch_text(download_url), config.upstream_name)

        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.suffix == ".csproj":
            merged_content = merge_csproj_content(normalized_content, target_path)
            target_path.write_text(merged_content, encoding="utf-8", newline="\n")
        else:
            target_path.write_text(normalized_content, encoding="utf-8", newline="\n")

        expected_paths.append(target_path)

    expected_path_set = set(expected_paths)
    for existing_path in config.target_dir.rglob("*"):
        if existing_path.suffix not in ALLOWED_SUFFIXES:
            continue
        if existing_path not in expected_path_set:
            existing_path.unlink()

    for existing_dir in sorted((path for path in config.target_dir.rglob("*") if path.is_dir()), reverse=True):
        if not any(existing_dir.iterdir()):
            existing_dir.rmdir()

    return expected_paths


def main() -> None:
    synced_paths: list[Path] = []
    for sample in SAMPLES:
        synced_paths.extend(sync_sample(sample))

    for path in sorted(synced_paths):
        print(path.relative_to(REPO_ROOT).as_posix())


if __name__ == "__main__":
    main()
