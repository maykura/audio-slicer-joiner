import numpy as np
import numpy.typing as npt
from sortedcontainers import SortedKeyList
import os.path
from tqdm.auto import tqdm
import librosa
import soundfile
import warnings


# This function is obtained from librosa.
def get_rms(
    y,
    *,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    # get_rms only works with floating point audio scaled between -1 to 1
    # convert the input audio to that if it is integer audio
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32, order='C') / max(abs(np.iinfo(y.dtype).max), abs(np.iinfo(y.dtype).min))
        # (numpy will automatically cast to higher precision if needed to prevent information loss)

    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    # put our new within-frame axis at the end for now
    out_strides = y.strides + tuple([y.strides[axis]])
    # Reduce the shape on the framing axis
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(
        y, shape=out_shape, strides=out_strides
    )
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    return np.sqrt(power)


def get_bit_depth(subtype: str) -> npt.DTypeLike:
    match subtype:
        case 'PCM_16' | 'PCM_S8' | 'ALAC_16':
            return np.int16
        case 'PCM_32':
            return np.int32
        case 'DOUBLE':
            return np.float64
        case _:
            return np.float32


def get_subtype(dtype: npt.DTypeLike) -> str:
    match dtype:
        case np.int16:
            return 'PCM_16'
        case np.int32:
            return 'PCM_32'
        case np.float64:
            return 'DOUBLE'
        case _:
            return 'FLOAT'


def upgrade_channels(waveform: np.array, out_channels: int) -> np.array:
    in_channels = len(waveform.shape)
    if in_channels >= out_channels:
        return waveform
    elif in_channels > 1:
        raise ValueError("Only mono-to-multi channels conversion is supported")

    output = np.repeat(waveform[:, np.newaxis], out_channels, axis=1)
    return output


def compare_bitdepth(left: npt.DTypeLike, right: npt.DTypeLike) -> npt.DTypeLike:
    """
    Compares two numpy datatypes and return the one that the other can be safely cast to.
    Returns float64 as default if neither operand can be cast to the other.
    """
    if left != right and not (np.dtype(left) < np.dtype(right)) and not (np.dtype(left) > np.dtype(right)):
        return np.float64
    else:
        return max(np.dtype(left), np.dtype(right))


class Slicer:
    def __init__(self,
                 sr: int,
                 threshold: float = -40.,
                 min_length: int = 5000,
                 min_interval: int = 300,
                 hop_size: int = 20,
                 max_sil_kept: int = 5000):
        if not min_length >= min_interval >= hop_size:
            raise ValueError('The following condition must be satisfied: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('The following condition must be satisfied: max_sil_kept >= hop_size')
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]

    def find_slices(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform

        length = (samples.shape[0] + self.hop_size - 1) // self.hop_size
        if length <= self.min_length:
            return [(0, length)]

        rms_list = get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        # Apply and return slices.
        if len(sil_tags) == 0:
            return [(0, length)]
        else:
            slices = []
            if sil_tags[0][0] > 0:
                slices.append((0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                slices.append((sil_tags[i][1], sil_tags[i + 1][0]))
            if sil_tags[-1][1] < total_frames:
                slices.append((sil_tags[-1][1], total_frames))

            return slices

    # @timeit
    def slice(self, waveform):
        chunks = []
        for chunk in self.find_slices(waveform):
            chunks.append(self._apply_slice(waveform, chunk[0], chunk[1]))
        return chunks


class Joiner:
    """
    Provides an interface for building an index to keep track of audio slices that can then be joined together

    Attributes:
        index (SortedKeyList): The index to keep track of all the audio slices.
            Implemented as a sortedcontainers.SortedKeyList with the following keys::
                - path: full path to the audio file containing the slice.
                - length: length of the slice, in milliseconds.
                - start: start time of the slice within the audio file, in milliseconds.
    """
    def __init__(self, index: SortedKeyList = SortedKeyList(key=lambda d: d['length'])):
        self.index = index

    @property
    def index(self) -> SortedKeyList:
        return self._index

    @index.setter
    def index(self, value: SortedKeyList):
        if type(value) != SortedKeyList:
            raise ValueError("index must be a SortedKeyList!")
        self._index = value

    def append_index(self, path: str, length: int, start: int):
        self._index.add({'path': path, 'length': length, 'start': start})

    @staticmethod
    def create_index_from_files(in_path: str, rejoin_length: int) -> SortedKeyList:
        """
        Returns a SortedKeyList that indexes each audio file from the given directory as a single slice.

        NOTE that this simply creates and returns a new index without modifying any Joiner object's index attribute
        """
        if rejoin_length <= 0:
            raise ValueError("Rejoin_length is zero or less. Cannot create index")

        index = SortedKeyList(key=lambda d: d['length'])
        paths = []
        if os.path.isfile(in_path):
            paths.append(in_path)
        elif os.path.isdir(in_path):
            for root, dirs, files in os.walk(in_path):
                for file in files:
                    paths.append(os.path.join(root, file))
        else:
            raise ValueError("Invalid path")

        for path in paths:
            sf = soundfile.SoundFile(path)
            index.add({'path': path, 'length': sf.frames / sf.samplerate * 1000, 'start': 0})

        return index

    def join(self, rejoin_length: int, out_path: str, in_path: str = None):
        """
        Args:
            rejoin_length: Maximum duration of each file containing the joined segments, in milliseconds.
            out_path: Output directory for the files containing the joined segments.
                Path will be created if it doesn't exist already.
            in_path: If specified, each audio file in the specified directory will be considered as a single slice
                and be used for joining, and the index attribute will be ignored.
                If not specified, the index attribute will be used for joining.
        """
        if rejoin_length <= 0:
            warnings.warn("Rejoin_length is zero or less. Nothing to do")
            return
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # Build index from files if in_path is specified, otherwise copy the index property
        if in_path is not None:
            index = self.create_index_from_files(in_path, rejoin_length)
        else:
            index = self._index.copy()

        if len(index) <= 0:
            warnings.warn("Index is empty, nothing to do")
            return

        out_files = []
        # Iterate through the index from the longest segment and join it with shorter segments to fill the gap until rejoin_length is met
        while len(index) > 0:
            segment = index.pop()
            out_files.append([segment])

            gap = rejoin_length - segment['length']
            if segment['length'] >= rejoin_length:
                continue

            # Keep finding segments that fit in the gap and join them
            # until not even the shortest segment left in the index can fit in the gap
            while len(index) > 0 and gap >= index[0]['length']:
                # Use bisect_key_left() to find the index position of the segment whose duration matches exactly to the gap
                # If there is no such segment, bisect_key_left() would return the position of the first segment that exceeds the gap
                # Subtract 1 to get the position of longest segment that fits in the gap
                insert_i = index.bisect_key_left(gap)
                if insert_i < len(index) and index[insert_i]['length'] <= gap:
                    insert = index.pop(insert_i)
                else:
                    insert = index.pop(insert_i-1)

                out_files[-1].append(insert)
                gap -= insert['length']

        for file in tqdm(out_files, desc="Re-joining and writing to files..."):
            max_channels = 0
            max_samplerate = 0
            # Set to a minimum bit depth of int16 since that's the lowest bit depth this script would accept
            max_bitdepth = np.dtype(np.int16)

            # Iterate through all the segments in the file once first to determine the max sample rate, bit depth,
            # and channels count that all segments should be converted to later
            for seg in file:
                sf = soundfile.SoundFile(seg['path'])
                # Bit depth must be at least float32 if resampling is needed
                if max_samplerate != 0 and max_samplerate != sf.samplerate:
                    max_bitdepth = compare_bitdepth(np.dtype(np.float32), compare_bitdepth(max_bitdepth, np.dtype(get_bit_depth(sf.subtype))))
                else:
                    max_bitdepth = compare_bitdepth(max_bitdepth, np.dtype(get_bit_depth(sf.subtype)))

                max_channels = max(max_channels, sf.channels)
                max_samplerate = max(max_samplerate, sf.samplerate)
                sf.close()

            chunk_list = []
            filename = ""
            for seg in file:
                # Set sr to None if resampling is not needed.
                # When a sr value other None is passed to librosa it would always convert the audio to float32 even if there is no need of resampling
                sr = max_samplerate if librosa.get_samplerate(seg['path']) != max_samplerate else None
                chunk, _ = librosa.load(seg['path'], sr=sr, mono=False, dtype=max_bitdepth, offset=(seg['start']/1000), duration=(seg['length']/1000))
                if len(chunk.shape) > 1:
                    chunk = chunk.T
                chunk = upgrade_channels(chunk, max_channels)
                chunk_list += chunk.tolist()
                filename += os.path.basename(seg['path']).rsplit('.', maxsplit=1)[0] + f"_{seg['start']//1000}-"

            # For the filename, first remove the trailing delimiter character (-)
            # Then to account for the maximum length for a path in Windows, which is 259 characters, subtract ".wav" = 255
            # Then subtract the directory path to get the max filename length
            soundfile.write(file=os.path.join(out_path, f'%s.wav' % filename[:-1][:255-len(out_path)]),
                            data=np.array(chunk_list, dtype=max_bitdepth),
                            samplerate=max_samplerate,
                            subtype=get_subtype(max_bitdepth))


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('audio', type=str, help='The audio file or path containing audio files to be sliced')
    parser.add_argument('--out', type=str, help='Output directory of the sliced audio clips')
    parser.add_argument('--db_thresh', type=float, required=False, default=-40,
                        help='The dB threshold for silence detection')
    parser.add_argument('--min_length', type=int, required=False, default=5000,
                        help='The minimum milliseconds required for each sliced audio clip')
    parser.add_argument('--min_interval', type=int, required=False, default=300,
                        help='The minimum milliseconds for a silence part to be sliced')
    parser.add_argument('--hop_size', type=int, required=False, default=10,
                        help='Frame length in milliseconds')
    parser.add_argument('--max_sil_kept', type=int, required=False, default=500,
                        help='The maximum silence length kept around the sliced clip, presented in milliseconds')
    parser.add_argument('--rejoin_length', type=int, required=False, default=-1,
                        help='Re-join slices into segments of length in milliseconds. Values below 1 disable rejoining')
    args = parser.parse_args()
    out = args.out
    if out is None:
        out = os.path.dirname(os.path.abspath(args.audio))
    elif not os.path.exists(out):
        os.makedirs(out)

    audio_paths = []
    if os.path.isfile(args.audio):
        audio_paths.append(args.audio)
    elif os.path.isdir(args.audio):
        for root, dirs, files in os.walk(args.audio):
            for file in files:
                audio_paths.append(os.path.join(root, file))

    joiner = Joiner()

    for path in tqdm(audio_paths, desc="Slicing..."):
        try:
            with soundfile.SoundFile(path) as file:
                audio, sr = librosa.load(path, sr=None, mono=False, dtype=get_bit_depth(file.subtype))
        except:
            print(f"Error loading {path} as an audio file")
            continue
        slicer = Slicer(
            sr=sr,
            threshold=args.db_thresh,
            min_length=args.min_length,
            min_interval=args.min_interval,
            hop_size=args.hop_size,
            max_sil_kept=args.max_sil_kept
        )
        if args.rejoin_length > 0:
            chunks = slicer.find_slices(audio)
            for chunk in chunks:
                joiner.append_index(path, (chunk[1] - chunk[0]) * args.hop_size, chunk[0] * args.hop_size)
        else:
            chunks = slicer.slice(audio)
            for i, chunk in enumerate(chunks):
                if len(chunk.shape) > 1:
                    chunk = chunk.T
                soundfile.write(
                    file=os.path.join(out, f'%s_%d.wav' % (os.path.basename(path).rsplit('.', maxsplit=1)[0], i)),
                    data=chunk,
                    samplerate=sr,
                    subtype=get_subtype(chunk.dtype))

    if args.rejoin_length > 0:
        joiner.join(rejoin_length=args.rejoin_length, out_path=out)

if __name__ == '__main__':
    main()
