"""Main module for camsort package."""

##### IMPORTS #####

import abc
import collections
import dataclasses
import datetime
import fractions
import functools
import itertools
import json
import logging
import os
import pathlib
import warnings
from collections.abc import Collection, Mapping
from typing import Any, Literal, NamedTuple

import contextily as cx
import numpy as np
import pandas as pd
from matplotlib import axes, dates
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
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

    return float(lat), float(lon)


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
        data["location"] = self.location._asdict()
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

    @property
    def created_at(self) -> datetime.datetime:
        """Created date / time for the media file."""
        tags = self.load_tags()
        try:
            return datetime.datetime.strptime(
                tags[ExifTags.Base.DateTime], "%Y:%m:%d %H:%M:%S"
            )
        except KeyError:
            return datetime.datetime.fromtimestamp(self.filestats.st_mtime)

    @functools.cached_property
    def location(self) -> Location:
        """Image location as longitude, latitude and altitude."""
        gps = self.load_tags()[ExifTags.Base.GPSInfo]

        lat = ExifTags.GPSTAGS[ExifTags.GPS.GPSLatitude]
        lon = ExifTags.GPSTAGS[ExifTags.GPS.GPSLongitude]
        lat_ref = ExifTags.GPSTAGS[ExifTags.GPS.GPSLatitudeRef]
        lon_ref = ExifTags.GPSTAGS[ExifTags.GPS.GPSLongitudeRef]

        try:
            latitude, longitude = degrees_minutes_to_decimal(
                gps[lat], gps[lon], gps[lat_ref], gps[lon_ref]
            )
        except KeyError:
            latitude, longitude = np.nan, np.nan

        return Location(longitude, latitude, gps.get(ExifTags.GPS.GPSAltitude, np.nan))


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
    if isinstance(obj, fractions.Fraction):
        return str(obj)
    raise TypeError(f"{type(obj)} not JSON serializable")


def dump_json(path: pathlib.Path, obj: object) -> None:
    """Dump `obj` to JSON file."""
    LOG.info("Writing JSON")
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=4, default=_json_encoder)
    LOG.info("Written: %s", path)


def find_media_files(base_folder: pathlib.Path) -> dict[str, list[MediaFile]]:
    """Find images and videos in `base_folder` and sub-folders."""
    LOG.info("Finding media within %s", base_folder)
    media_files: dict[str, list[MediaFile]] = collections.defaultdict(list)

    count = 0
    count_other = 0
    for folder, _, filenames in base_folder.walk():
        for name in filenames:
            path = folder / name
            if path.suffix.lower() in _IMAGE_SUFFIXES:
                media_class = ImageFile
            elif path.suffix.lower() in _VIDEO_SUFFIXES:
                media_class = MediaFile
            else:
                count_other += 1
                continue

            extras = [i for i in filenames if i.startswith(name) and i != name]
            if len(extras) == 0:
                media = media_class(path)
            else:
                media = media_class(path, extras=extras)

            media_files[name].append(media)
            count += 1

    LOG.info(
        "Finished searching: %s media files found and %s others",
        f"{count:,}",
        f"{count_other:,}",
    )
    return dict(media_files)


def plot_images(images: list[ImageFile], output_path: pathlib.Path) -> None:
    """Plot image locations and dates."""
    data = pd.DataFrame(
        [
            (i.path.stem, i.location.latitude, i.location.longitude, i.created_at)
            for i in images
        ],
        columns=["name", "latitude", "longitude", "datetime"],
    )
    output_path = output_path.with_suffix(".csv")
    data.to_csv(output_path, index=False)
    LOG.info("Written: %s", output_path)

    location_mask = ~data[["latitude", "longitude"]].isna().any(axis=1)
    date_mask = ~data["datetime"].isna()

    LOG.info(
        "%s (%s) have location, %s (%s) have a timestamp and %s (%s) have both",
        f"{location_mask.sum():,}",
        f"{location_mask.sum() / len(data):.1%}",
        f"{date_mask.sum():,}",
        f"{date_mask.sum() / len(data):.1%}",
        f"{(location_mask & date_mask).sum():,}",
        f"{(location_mask & date_mask).sum() / len(data):.1%}",
    )

    # TODO: calculate groups using sklearn clustering
    rng = np.random.default_rng()
    data["group"] = rng.integers(0, 10, len(data))

    fig = plt.figure(figsize=(15, 15))
    nrows, ncols = 2, 2
    _map_locations(
        fig.add_subplot(
            nrows,
            ncols,
            (1, 2),
        ),
        data,
        location_mask,
    )
    _plot_timeline(fig.add_subplot(nrows, ncols, 3), data, date_mask)
    _plot_3d(
        fig.add_subplot(nrows, ncols, 4, projection="3d"), data, location_mask & date_mask
    )

    fig.set_layout_engine("tight")
    output_path = output_path.with_suffix(".pdf")
    fig.savefig(output_path)
    LOG.info("Written: %s", output_path)


def _plot_3d(ax: mplot3d.Axes3D, data: pd.DataFrame, mask: pd.Series) -> None:
    scatter = ax.scatter(
        data.loc[mask, "longitude"],
        data.loc[mask, "latitude"],
        dates.date2num(data.loc[mask, "datetime"]),
        c=data.loc[mask, "group"],
    )
    ax.legend(*scatter.legend_elements(), title="Random Clusters")

    ax.set_title("Images by Location and Timestamp")
    ax.set_zlabel("Datetime")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def _map_locations(
    ax: axes.Axes, data: pd.DataFrame, mask: pd.Series, margin: float = 0.2
) -> None:
    limits = {}
    for col in ("longitude", "latitude"):
        min_ = data.loc[mask, col].min()
        max_ = data.loc[mask, col].max()

        diff = abs(max_ - min_)
        limits[col] = (min_ - (diff * margin), max_ + (diff * margin))

    # Make Y min diff at least 60% of x min diff so the graph fills more space
    ratio = 0.6
    y_diff = abs(limits["latitude"][1] - limits["latitude"][0])
    x_diff = abs(limits["longitude"][1] - limits["longitude"][0])
    if y_diff < (ratio * x_diff):
        adjustment = (ratio * x_diff) - y_diff
        limits["latitude"] = (
            limits["latitude"][0] - (adjustment // 2),
            limits["latitude"][1] + (adjustment // 2),
        )

    scatter = ax.scatter(
        data.loc[mask, "longitude"],
        data.loc[mask, "latitude"],
        c=data.loc[mask, "group"],
    )
    ax.legend(*scatter.legend_elements(), title="Random Cluster")
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(limits["longitude"])
    ax.set_ylim(limits["latitude"])
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, crs="EPSG:4326")
    ax.set_title("Map of Image Locations")


def _plot_timeline(ax: axes.Axes, data: pd.DataFrame, mask: pd.Series) -> None:
    labels = []
    datasets = []
    for group, group_data in data.loc[mask, :].groupby("group")["datetime"]:
        labels.append(group)
        datasets.append(dates.date2num(group_data.to_numpy()))

    ax.hist(datasets, bins=30, histtype="barstacked", label=labels)
    ax.legend(title="Random Cluster")

    ax.set_title("Images by Timestamp")
    locator = dates.YearLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(dates.AutoDateFormatter(locator))
    ax.xaxis.set_tick_params("major", rotation=45)


def main():
    base_folder = pathlib.Path("data/images")
    output_folder = base_folder.parent / "camsort-outputs"
    output_folder.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    media = find_media_files(base_folder)
    # dump_json(output_folder / "media.json", media)

    plot_images(
        [i for i in itertools.chain.from_iterable(media.values()) if isinstance(i, ImageFile)],
        output_folder / "image_locations",
    )

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

    dump_json(output_folder / "others.json", others)


if __name__ == "__main__":
    main()
