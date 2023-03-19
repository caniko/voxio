from pydantic import BaseModel, DirectoryPath, FilePath


class CachingInfo(BaseModel):
    data_directory: DirectoryPath

    @property
    def cache_directory(self) -> DirectoryPath:
        result = self.data_directory / "cache"
        result.mkdir(exist_ok=True)
        return result

    @property
    def state_path(self) -> FilePath:
        return self.cache_directory / "state.pickle.lz4"

    @property
    def array_directory(
        self,
    ) -> DirectoryPath:
        result = self.cache_directory / "array"
        result.mkdir(parents=True, exist_ok=True)
        return result

    @property
    def array_file_paths(self) -> tuple[FilePath, ...]:
        return tuple(list(self.array_directory.iterdir()))

    @property
    def number_of_arrays(self) -> int:
        return len(self.array_file_paths)

    def volume_array_by_index(self, volume_index: int) -> FilePath:
        return self.array_directory / f"{volume_index}.npz"
