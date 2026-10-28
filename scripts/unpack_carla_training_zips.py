"""Unpack the downloaded CARLA route archives from the LEAD project root."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile


def unpack(path: Path) -> None:
    with ZipFile(path) as archive:
        archive.extractall(Path.cwd())


def main() -> None:
    archives = sorted(Path("data/carla_leaderboard2/zip").glob("*/*.zip"))
    if not archives:
        raise SystemExit("No route archives found in data/carla_leaderboard2/zip")
    print(f"Unpacking {len(archives)} route archives...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for count, _ in enumerate(executor.map(unpack, archives), 1):
            if count % 100 == 0 or count == len(archives):
                print(f"{count}/{len(archives)}", flush=True)


if __name__ == "__main__":
    main()
