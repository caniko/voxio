from pathlib import Path

from pydantic import FilePath

FilePathLike = FilePath | Path | str
TupleSlice = tuple[int, int]
