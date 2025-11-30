"""Main module for camsort package."""

##### IMPORTS #####

import abc
import collections
import dataclasses
import datetime
import functools
import itertools
import json
import logging
import os
import pathlib
import warnings
from collections.abc import Collection, Mapping
from typing import Any, Literal, NamedTuple

from PIL import ExifTags, Image

##### CONSTANTS #####

LOG = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".png"}
_VIDEO_SUFFIXES = {".mp4", ".mp"}


##### CLASSES & FUNCTIONS #####

_BasicScalars = int | float | str
_Serializable = (
    _BasicScalars | Collection[_BasicScalars] | Mapping[_BasicScalars, _BasicScalars]
)


def degrees_minutes_to_decimal(
    latitude: tuple[float, float, float],
    longitude: tuple[float, float, float],
    latitude_ref: Literal["N", "S"],
    longitude_ref: Literal["E", "W"],
) -> tuple[float, float]:
    """Convert latitude / longitude from minutes and seconds to decimal degrees."""
    lat = latitude[0] + (latitude[1] / 60) + (latitude[2] / 3600)
    lon = longitude[0] + (longitude[1] / 60) + (longitude[2] / 3600)
    if latitude_ref == "S":
        lat *= -1
    if longitude_ref == "W":
        lon *= -1

    return lat, lon


class Location(NamedTuple):
    """Image location coordinates and altitude."""

    longitude: float
    latitude: float
    altitude: float


def _basic_types(value: Any) -> _Serializable:
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, Mapping):
        return type(value)({i: _basic_types(j) for i, j in value.items()})
    if isinstance(value, Collection):
        return type(value)(_basic_types(i) for i in value)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()

    try:
        int(value)
    except ValueError:
        return str(value)

    return int(value) if int(value) == float(value) else float(value)


class JSONSerializable(abc.ABC):
    """Class containing a method for serializing to valid JSON types."""

    @abc.abstractmethod
    def __json__(self) -> _Serializable:
        """Serialize class to acceptable JSON types."""
        raise NotImplementedError("Abstract method")


@dataclasses.dataclass
class MediaFile(JSONSerializable):
    """Metadata for media files."""

    path: pathlib.Path
    extras: list[str] | None = None

    _metadata: str | None = None

    def __post_init__(self) -> None:
        """Find JSON metadata in extras."""
        if self.extras is None:
            return

        json_files = [i for i in self.extras if i.endswith(".json")]
        if len(json_files) > 0:
            self._metadata = json_files[0]
            if len(json_files) > 1:
                warnings.warn(
                    f"found {len(json_files)} JSON metadata? files for {self.path.name}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def __json__(self) -> _Serializable:
        """Serialize class to acceptable JSON types."""
        data = dataclasses.asdict(self)
        if self._metadata is not None:
            data["metadata"] = self.load_metadata()
        data["dates"] = self.dates()
        return data

    @property
    def paths(self) -> list[pathlib.Path]:
        """Paths to media file and any supplementary files."""
        paths = [self.path]
        if self.extras is not None:
            paths.extend(self.path.with_name(i) for i in self.extras)

        return paths

    def load_metadata(self) -> dict:
        """Load supplementary metadata file."""
        if self._metadata is None:
            raise ValueError("no supplementary file")
        path = self.path.with_name(self._metadata)
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @functools.cached_property
    def filestats(self) -> os.stat_result:
        """OS file stats for the media path."""
        return self.path.stat()

    @property
    def created_at(self) -> datetime.datetime:
        """Created date / time for the media file."""
        return datetime.datetime.fromtimestamp(self.filestats.st_birthtime)

    def dates(self) -> dict[str, datetime.datetime]:
        """Get created and modified dates from file stats."""
        return {
            "created": self.created_at,
            "modified": datetime.datetime.fromtimestamp(self.filestats.st_mtime),
        }


class ImageFile(MediaFile):
    """Metadata for image file."""

    def __json__(self) -> dict:
        """Serialize class to acceptable JSON types."""
        data = super().__json__()

        tags = self.load_tags()
        # Attempt to convert tags to JSON serializable types
        data["exif"] = {ExifTags.TAGS[i]: _basic_types(j) for i, j in tags.items()}
        return data

    def load_tags(self) -> dict[int, _Serializable]:
        """Load exif tags for image."""
        with Image.open(self.path) as image:
            exif = image.getexif()
            tags = dict(exif.items())
            tags[ExifTags.Base.GPSInfo] = {
                ExifTags.GPSTAGS[i]: j for i, j in exif.get_ifd(ExifTags.Base.GPSInfo).items()
            }

        return tags

    @functools.cached_property
    def location(self) -> Location:
        """Image location as longitude, latitude and altitude."""
        gps = self.load_tags()[ExifTags.Base.GPSInfo]
        latitude, longitude = degrees_minutes_to_decimal(
            gps[ExifTags.GPS.GPSDestLatitude],
            gps[ExifTags.GPS.GPSDestLongitude],
            gps[ExifTags.GPS.GPSDestLatitudeRef],
            gps[ExifTags.GPS.GPSDestLongitudeRef],
        )
        return Location(
            latitude,
            longitude,
            gps[ExifTags.GPS.GPSAltitude],
        )


def _json_encoder(
    obj: JSONSerializable | pathlib.Path | datetime.datetime | datetime.date,
) -> _Serializable:
    if hasattr(obj, "__json__"):
        return obj.__json__()
    if isinstance(obj, pathlib.Path):
        return str(obj.resolve())
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode()
    raise TypeError(f"{type(obj)} not JSON serializable")


def dump_json(path: pathlib.Path, obj: object) -> None:
    """Dump `obj` to JSON file."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=4, default=_json_encoder)
    LOG.info("Written: %s", path)


def find_media_files(base_folder: pathlib.Path) -> dict[str, list[MediaFile]]:
    """Find images and videos in `base_folder` and sub-folders."""
    media_files: dict[str, list[MediaFile]] = collections.defaultdict(list)

    for folder, _, filenames in base_folder.walk():
        for name in filenames:
            path = folder / name
            if path.suffix.lower() in _IMAGE_SUFFIXES:
                media_class = ImageFile
            elif path.suffix.lower() in _VIDEO_SUFFIXES:
                media_class = MediaFile
            else:
                continue

            extras = [i for i in filenames if i.startswith(name) and i != name]
            if len(extras) == 0:
                media = media_class(path)
            else:
                media = media_class(path, extras=extras)
            media_files[name].append(media)

    return dict(media_files)


def main():
    base_folder = pathlib.Path()
    logging.basicConfig(level=logging.INFO)

    media = find_media_files(base_folder)

    dump_json(pathlib.Path("media.json"), media)

    found_paths = list(
        itertools.chain.from_iterable(
            i.paths for i in itertools.chain.from_iterable(media.values())
        )
    )
    others: dict[str, list[pathlib.Path]] = collections.defaultdict(list)
    for folder, _, filenames in base_folder.walk():
        for name in filenames:
            path = folder / name
            if path.suffix.lower() in (_IMAGE_SUFFIXES | _VIDEO_SUFFIXES):
                continue

            if path not in found_paths:
                others[path.suffix].append(path)

    dump_json(pathlib.Path("others.json"), others)


if __name__ == "__main__":
    main()
