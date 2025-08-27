#!/usr/bin/env python3
import errno
import multiprocessing
import sys
import os
import os.path as path
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
from typing import Any, Callable, Optional
import logging
import mutagen.flac
from . import tagging

LOGGER = logging.getLogger("transcode")

encoders = {
    "320": {"enc": "lame", "ext": ".mp3", "opts": "-h -b 320 --ignore-tag-errors"},
    "V0": {"enc": "lame", "ext": ".mp3", "opts": "-V 0 --vbr-new --ignore-tag-errors"},
    "V2": {"enc": "lame", "ext": ".mp3", "opts": "-V 2 --vbr-new --ignore-tag-errors"},
    "FLAC": {"enc": "flac", "ext": ".flac", "opts": "--best"},
}


class TranscodeException(Exception):
    pass


class TranscodeDownmixException(TranscodeException):
    pass


class UnknownSampleRateException(TranscodeException):
    pass


class ArtistAlbumMismatchException(TranscodeException):
    pass


def normalize_string_for_comparison(s):
    """Normalize string for case-insensitive comparison by removing special chars and lowercasing"""
    if not s:
        return ""
    # Remove common special characters and normalize whitespace
    normalized = re.sub(r'[^\w\s]', '', s.strip().lower())
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def validate_artist_album_match(flac_dir, group_info=None, torrent_info=None):
    """
    Validate that the artist and album in the FLAC files match the expected values
    from the API data or directory structure.
    
    Args:
        flac_dir: Directory containing FLAC files
        group_info: Group information from API (optional)
        torrent_info: Torrent information from API (optional)
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Get expected artist and album from API data if available
        expected_artist = None
        expected_album = None
        
        if group_info and torrent_info:
            group = group_info.get("group", {})
            expected_artist = (
                group.get("musicInfo", {})
                .get("artists", [{}])[0]
                .get("name", "")
            )
            expected_album = group.get("name", "")
            
        # If no API data, try to extract from directory name
        if not expected_artist or not expected_album:
            LOGGER.info("No API data available, extracting from directory name")
            dir_name = Path(flac_dir).name
            # Try to parse directory name like "Artist - Album (Year) [extra info]"
            match = re.match(r'^(.+?)\s*-\s*(.+?)(?:\s*\(\d{4}\))?(?:\s*\[.+?\])?(?:\s*\{.+?\})?$', dir_name)
            if match:
                expected_artist = match.group(1).strip()
                expected_album = match.group(2).strip()
            else:
                LOGGER.warning("Could not parse artist/album from directory name, skipping validation")
                return True, None
        
        if not expected_artist or not expected_album:
            LOGGER.warning("No artist/album information available for validation")
            return True, None
            
        # Normalize expected values
        expected_artist_norm = normalize_string_for_comparison(expected_artist)
        expected_album_norm = normalize_string_for_comparison(expected_album)
        
        LOGGER.info(f"Validating files against expected artist: '{expected_artist}', album: '{expected_album}'")
        
        # Check a few FLAC files to validate artist/album
        flac_files = list(locate(flac_dir, ext_matcher(".flac")))
        if not flac_files:
            return False, "No FLAC files found in directory"
            
        # Sample up to 3 files for validation (don't need to check every file)
        files_to_check = flac_files[:min(3, len(flac_files))]
        
        for flac_file in files_to_check:
            try:
                flac_info = mutagen.flac.FLAC(flac_file)
                
                # Get artist from file tags
                file_artist = ""
                if "artist" in flac_info:
                    file_artist = flac_info["artist"][0] if flac_info["artist"] else ""
                elif "albumartist" in flac_info:
                    file_artist = flac_info["albumartist"][0] if flac_info["albumartist"] else ""
                
                # Get album from file tags  
                file_album = ""
                if "album" in flac_info:
                    file_album = flac_info["album"][0] if flac_info["album"] else ""
                
                # Normalize file values
                file_artist_norm = normalize_string_for_comparison(file_artist)
                file_album_norm = normalize_string_for_comparison(file_album)
                
                # Compare normalized values
                artist_match = file_artist_norm == expected_artist_norm
                album_match = file_album_norm == expected_album_norm
                
                if not artist_match or not album_match:
                    error_parts = []
                    if not artist_match:
                        error_parts.append(f"Artist mismatch - Expected: '{expected_artist}', Found: '{file_artist}'")
                    if not album_match:
                        error_parts.append(f"Album mismatch - Expected: '{expected_album}', Found: '{file_album}'")
                    
                    error_msg = f"File '{os.path.basename(flac_file)}': {'; '.join(error_parts)}"
                    LOGGER.error(    error_msg)
                    return False, error_msg
                    
            except Exception as e:
                LOGGER.warning(f"Could not read tags from {flac_file}: {e}")
                continue
                
        LOGGER.info("    Artist/album validation passed")
        return True, None
        
    except Exception as e:
        LOGGER.error(f"    Error during artist/album validation: {e}")
        return False, f"    Validation error: {e}"


# In most Unix shells, pipelines only report the return code of the
# last process. We need to know if any process in the transcode
# pipeline fails, not just the last one.
#
# This function constructs a pipeline of processes from a chain of
# commands just like a shell does, but it returns the status code (and
# stderr) of every process in the pipeline, not just the last one. The
# results are returned as a list of (code, stderr) pairs, one pair per
# process.
def run_pipeline(cmds):
    # The Python executable (and its children) ignore SIGPIPE. (See
    # http://bugs.python.org/issue1652) Our subprocesses need to see
    # it.
    sigpipe_handler = signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    stdin = None
    last_proc = None
    procs = []
    try:
        for cmd in cmds:
            proc = subprocess.Popen(
                shlex.split(cmd),
                stdin=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if last_proc is not None and last_proc.stdout is not None:
                # Ensure last_proc receives SIGPIPE if proc exits first
                last_proc.stdout.close()
            procs.append(proc)
            stdin = proc.stdout
            last_proc = proc
    finally:
        signal.signal(signal.SIGPIPE, sigpipe_handler)

    # Handle empty command list case
    if last_proc is None:
        return []

    last_stderr = last_proc.communicate()[1]

    results = []
    for cmd, proc in zip(cmds[:-1], procs[:-1]):
        # wait() is OK here, despite use of PIPE above; these procs
        # are finished.
        proc.wait()
        results.append((proc.returncode, proc.stderr.read()))
    results.append((last_proc.returncode, last_stderr))
    return results


def locate(root, match_function: Callable[[str], Any], ignore_dotfiles=True):
    """
    Yields all filenames within the root directory for which match_function returns True.
    """
    for dir_path, dirs, files in os.walk(root):
        for filename in (
            os.path.abspath(os.path.join(dir_path, filename))
            for filename in files
            if match_function(filename)
        ):
            if ignore_dotfiles and os.path.basename(filename).startswith("."):
                pass
            else:
                yield filename


def ext_matcher(*extensions: str) -> Callable[[str], bool]:
    """
    Returns a function which checks if a filename has one of the specified extensions. Expects a string in the format '.ext'.
    """
    return lambda f: os.path.splitext(f)[-1].lower() in extensions


def is_24bit(flac_dir):
    """
    Returns True if any FLAC within flac_dir is 24 bit.
    """
    flacs = (
        mutagen.flac.FLAC(flac_file)
        for flac_file in locate(flac_dir, ext_matcher(".flac"))
    )
    return any(flac.info.bits_per_sample > 16 for flac in flacs)


def is_multichannel(flac_dir):
    """
    Returns True if any FLAC within flac_dir is multichannel.
    """
    flacs = (
        mutagen.flac.FLAC(flac_file)
        for flac_file in locate(flac_dir, ext_matcher(".flac"))
    )
    return any(flac.info.channels > 2 for flac in flacs)


def needs_resampling(flac_dir):
    """
    Returns True if any FLAC within flac_dir needs resampling when
    transcoded.
    """
    return is_24bit(flac_dir)


def resample_rate(flac_dir):
    """
    Returns the rate to which the release should be resampled.
    """
    flacs = (
        mutagen.flac.FLAC(flac_file)
        for flac_file in locate(flac_dir, ext_matcher(".flac"))
    )
    original_rate = max(flac.info.sample_rate for flac in flacs)
    if original_rate % 44100 == 0:
        return 44100
    elif original_rate % 48000 == 0:
        return 48000
    else:
        return None


def transcode_commands(
    output_format,
    resample: bool,
    needed_sample_rate: Optional[int],
    flac_file,
    transcode_file,
):
    """
    Return a list of transcode steps (one command per list element),
    which can be used to create a transcode pipeline for flac_file ->
    transcode_file using the specified output_format, plus any
    resampling, if needed.
    """
    if resample:
        flac_decoder = "sox {FLAC} -G -b 16 -t wav - rate -v -L {SAMPLERATE} dither"
    else:
        flac_decoder = "flac -dcs -- {FLAC}"

    lame_encoder = "lame -S {OPTS} - {FILE}"
    flac_encoder = "flac {OPTS} -o {FILE} -"

    transcoding_steps = [flac_decoder]

    if encoders[output_format]["enc"] == "lame":
        transcoding_steps.append(lame_encoder)
    elif encoders[output_format]["enc"] == "flac":
        transcoding_steps.append(flac_encoder)

    transcode_args = {
        "FLAC": shlex.quote(flac_file),
        "FILE": shlex.quote(transcode_file),
        "OPTS": encoders[output_format]["opts"],
        "SAMPLERATE": needed_sample_rate,
    }

    if output_format == "FLAC" and resample:
        commands = [
            "sox {FLAC} -G -b 16 {FILE} rate -v -L {SAMPLERATE} dither".format(
                **transcode_args
            )
        ]
    else:
        commands = map(lambda cmd: cmd.format(**transcode_args), transcoding_steps)
    return commands


# Pool.map() can't pickle lambdas, so we need a helper function.
def pool_transcode(args):
    return transcode(*args)


def transcode(flac_file, output_dir, output_format):
    """
    Transcodes a FLAC file into another format.
    """
    # gather metadata from the flac file
    flac_info = mutagen.flac.FLAC(flac_file)
    sample_rate = flac_info.info.sample_rate
    bits_per_sample = flac_info.info.bits_per_sample
    resample: bool = sample_rate > 48000 or bits_per_sample > 16

    # if resampling isn't needed then needed_sample_rate will not be used.
    needed_sample_rate = None

    if resample:
        if sample_rate % 44100 == 0:
            needed_sample_rate = 44100
        elif sample_rate % 48000 == 0:
            needed_sample_rate = 48000
        else:
            raise UnknownSampleRateException(
                'FLAC file "{0}" has a sample rate {1}, which is not 88.2, 176.4, 96, or 192kHz but needs resampling, this is unsupported'.format(flac_file, sample_rate)
            )

    if flac_info.info.channels > 2:
        raise TranscodeDownmixException(
            'FLAC file "{0}" has more than 2 channels, unsupported'.format(flac_file)
        )

    # determine the new filename
    transcode_basename = path.splitext(os.path.basename(flac_file))[0]
    transcode_basename = re.sub(r'[\?<>\\*\|":]', "_", transcode_basename)
    transcode_file = path.join(output_dir, transcode_basename)
    transcode_file += encoders[output_format]["ext"]

    if not os.path.exists(path.dirname(transcode_file)):
        try:
            os.makedirs(path.dirname(transcode_file), exist_ok=True)
        except OSError as e:
            # Handle other OS errors that aren't about directory existing
            if e.errno != errno.EEXIST:
                raise e

    commands = list(
        transcode_commands(
            output_format, resample, needed_sample_rate, flac_file, transcode_file
        )
    )
    results = run_pipeline(commands)

    # Check for problems. Because it's a pipeline, the earliest one is
    # usually the source. The exception is -SIGPIPE, which is caused
    # by "backpressure" due to a later command failing: ignore those
    # unless no other problem is found.
    last_sigpipe = None
    for cmd, (code, stderr) in zip(commands, results):
        if code:
            if code == -signal.SIGPIPE:
                last_sigpipe = (cmd, (code, stderr))
            else:
                raise TranscodeException(
                    'Transcode of file "{0}" failed: {1}'.format(flac_file, stderr)
                )
    if last_sigpipe:
        # XXX: this should probably never happen....
        raise TranscodeException(
            'Transcode of file "{0}" failed: SIGPIPE'.format(flac_file)
        )

    tagging.copy_tags(flac_file, transcode_file)
    (ok, msg) = tagging.check_tags(transcode_file)
    if not ok:
        raise TranscodeException("Tag check failed on transcoded file: {0}".format(msg))

    return transcode_file


def get_transcode_dir(flac_dir, output_dir, output_format, resample, group_info=None, torrent_info=None) -> str:
    """
    Create a compliant directory name following the site rules:
    "Artist - Album (Year) [Catalog and edition info] {Media - Format}"
    
    Args:
        flac_dir: Source FLAC directory path
        output_dir: Output directory for transcoded files
        output_format: Target format (320, V0, V2, FLAC)
        resample: Whether resampling is needed
        group_info: Group information from API (optional, for proper naming)
        torrent_info: Torrent information from API (optional, for proper naming)
    """

    # Get the original directory name as fallback
    original_dir_name = Path(flac_dir).name
    
    # If we have API info, construct the proper name
    if group_info and torrent_info:
        group = group_info.get("group", {})
        artist = (
            group.get("musicInfo", {})
            .get("artists", [{}])[0]
            .get("name", "")
        )
        album = group.get("name", "")
        
        # For remastered releases, prefer torrent remaster info over group info
        if torrent_info.get('remastered'):
            # Use remasterYear if available, otherwise fall back to group year
            if torrent_info.get('remasterYear'):
                year = str(torrent_info.get('remasterYear'))
            else:
                year = str(group.get("year", ""))
        else:
            # For non-remastered releases, use group year
            year = str(group.get("year", ""))
        
        # If artist is not found in group_info, try torrent_info??? probs nothing there either
        if not artist:
            LOGGER.info("No artist found in group_info")
            LOGGER.info(f"group_info: {group_info}")
            LOGGER.info(f"torrent_info: {torrent_info}")
            sys.exit(1)
        
        # Build base name: "Artist - Album (Year)"
        base_name = f"{artist} - {album}"
        if year:
            base_name += f" ({year})"
            
        # Add edition info if this is a remaster
        edition_parts = []
        if torrent_info.get('remastered'):
            # Use remasterTitle if it exists and is not empty
            remaster_title = torrent_info.get('remasterTitle', '').strip()
            if remaster_title:
                edition_parts.append(remaster_title)

        # Add sample rate info for resampled releases
        if resample:
            rate = resample_rate(flac_dir)
            if rate == 44100:
                edition_parts.append("16-44.1")
            elif rate == 48000:
                edition_parts.append("16-48")
        
        # Combine edition info
        edition_info = ", ".join(edition_parts) if edition_parts else ""
        
        # Add media and format info
        media = torrent_info.get('media', 'CD')
                
        # Construct final name following the rules
        final_name = base_name
        if edition_info:
            final_name += f" [{edition_info}]"
        final_name += f" {{{media} - {output_format}}}"

    else:
        # Fallback to cleaning up the existing directory name
        final_name = original_dir_name
        
        # Remove common FLAC indicators and replace with target format
        flac_patterns = [
            r'\bFLAC\s*24[-\s]*BIT\b',
            r'\bFLAC[-\s]*24BIT\b', 
            r'\bFLAC[-\s]*24\b',
            r'\b24[-\s]*BIT\s*FLAC\b',
            r'\b24BIT\s*FLAC\b',
            r'\b24\s*FLAC\b',
            r'\bFLAC\s*HD\b',
            r'\bHD\s*FLAC\b',
            r'\bFLAC\b'
        ]
        
        format_replaced = False
        for pattern in flac_patterns:
            if re.search(pattern, final_name, re.IGNORECASE):
                # Map format for display
                format_display = {
                    "320": "320",
                    "V0": "V0",
                    "V2": "V2",
                    "FLAC": "FLAC",
                }.get(
                    output_format,
                    str(output_format) if output_format is not None else "",
                )

                final_name = re.sub(pattern, format_display, final_name, flags=re.IGNORECASE)
                format_replaced = True
                break
        
        # If no FLAC pattern was found, append format
        if not format_replaced:
            format_display = {
                '320': '320',
                'V0': 'V0',
                'V2': 'V2', 
                'FLAC': 'FLAC'
            }.get(output_format, output_format)
            
            # Check if we already have media-format notation
            if re.search(r'\{[^}]*\}$', final_name):
                # Replace existing format
                final_name = re.sub(r'\{[^}]*\}$', f'{{CD - {format_display}}}', final_name)
            else:
                # Add format notation
                final_name += f' {{CD - {format_display}}}'
        
        # Handle resampling info for fallback names
        if resample:
            rate = resample_rate(flac_dir)
            sample_info = ""
            if rate == 44100:
                sample_info = "16-44.1"
                # Replace high-res indicators
                final_name = re.sub(r'\b24[-\s]*176\.?4?\b', '16-44.1', final_name)
                final_name = re.sub(r'\b24[-\s]*88\.?2?\b', '16-44.1', final_name) 
                final_name = re.sub(r'\b24[-\s]*44\.?1?\b', '16-44.1', final_name)
                final_name = re.sub(r'\b24\b', '16', final_name)
            elif rate == 48000:
                sample_info = "16-48"
                # Replace high-res indicators  
                final_name = re.sub(r'\b24[-\s]*192\b', '16-48', final_name)
                final_name = re.sub(r'\b24[-\s]*96\b', '16-48', final_name)
                final_name = re.sub(r'\b24[-\s]*48\b', '16-48', final_name)
                final_name = re.sub(r'\b24\b', '16', final_name)
            
            # Add sample rate info if not already present
            if sample_info and sample_info not in final_name:
                # Insert before format notation if present
                if re.search(r'\{[^}]*\}$', final_name):
                    final_name = re.sub(r'\{', f'[{sample_info}] {{', final_name)
                else:
                    final_name += f' [{sample_info}]'
            LOGGER.info(f"final_name else: {final_name}")  # TODO
            sys.exit(1)  # TODO

    # Clean up the name
    # Remove invalid characters for file systems
    invalid_chars = r'[<>:"/\\|?*]'
    final_name = re.sub(invalid_chars, '_', final_name)
    
    # Remove leading/trailing spaces and limit length
    final_name = final_name.strip()
    
    # Ensure directory name isn't too long
    if len(final_name) > 200:  # Leave room for file paths
        final_name = final_name[:200].strip()
    
    # Remove any double spaces
    final_name = re.sub(r'\s+', ' ', final_name)

    return os.path.join(output_dir, final_name)


def transcode_release(
    flac_dir: str,
    output_dir: str,
    output_format: str,
    max_threads: Optional[int] = None,
    group_info: Optional[dict] = None,
    torrent_info: Optional[dict] = None,
):
    """
    Transcode a FLAC release into another format.
    """
    flac_dir = os.path.abspath(flac_dir)
    output_dir = os.path.abspath(output_dir)
    flac_files = locate(flac_dir, ext_matcher(".flac"))

    # Validate artist/album match before processing
    LOGGER.info("    Validating artist/album information...")
    is_valid, error_msg = validate_artist_album_match(flac_dir, group_info, torrent_info)
    if not is_valid:
        raise ArtistAlbumMismatchException(f"    Artist/album validation failed: {error_msg}")

    # check if we need to resample
    resample = needs_resampling(flac_dir)

    # check if we need to encode
    if output_format == "FLAC" and not resample:
        # XXX: if output_dir is not the same as flac_dir, this may not
        # do what the user expects.
        if output_dir != os.path.dirname(flac_dir):
            logging.info(
                "    Warning: no encode necessary, so files won't be placed in", output_dir
            )
        return flac_dir

    # make a new directory for the transcoded files
    #
    # NB: The cleanup code that follows this block assumes that
    # transcode_dir is a new directory created exclusively for this
    # transcode. Do not change this assumption without considering the
    # consequences!
    transcode_dir = get_transcode_dir(
        flac_dir, output_dir, output_format, resample, group_info, torrent_info
    )
    logging.info("    transcode_dir: " + transcode_dir)
    if not os.path.exists(transcode_dir):
        os.makedirs(transcode_dir)
    else:
        return transcode_dir
        # raise TranscodeException('transcode output directory "%s" already exists' % transcode_dir)

    try:
        arg_list = [
            (
                filename,
                path.dirname(filename).replace(flac_dir, transcode_dir),
                output_format,
            )
            for filename in flac_files
        ]
        for filename, output_dir, output_format in arg_list:
            transcode(filename, output_dir, output_format)
            LOGGER.info(f"Processing file {filename}")

        # copy other files
        allowed_extensions = [
            ".ac3",
            ".accurip",
            ".azw3",
            ".chm",
            ".cue",
            ".djv",
            ".djvu",
            ".doc",
            ".docx",
            ".dts",
            ".epub",
            ".ffp",
            ".flac",
            ".gif",
            ".htm",
            ".html",
            ".jpeg",
            ".jpg",
            ".json",
            ".lit",
            ".lrc",
            ".log",
            ".m3u",
            ".m3u8",
            ".m4a",
            ".m4b",
            ".md5",
            ".mobi",
            ".mp3",
            ".mp4",
            ".nfo",
            ".pdf",
            ".pls",
            ".png",
            ".rtf",
            ".sfv",
            ".txt",
            ".toc",
            ".yaml",
            ".yml",
        ]
        allowed_files = locate(flac_dir, ext_matcher(*allowed_extensions))
        for filename in allowed_files:
            new_dir = os.path.dirname(filename).replace(flac_dir, transcode_dir)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            shutil.copy(filename, new_dir)

        return transcode_dir

    except:
        # Cleanup.
        #
        # ASSERT: transcode_dir was created by this function and does
        # not contain anything other than the transcoded files!
        shutil.rmtree(transcode_dir)
        raise


def make_torrent(input_dir, output_dir, tracker, passkey, source):
    torrent = os.path.join(output_dir, path.basename(input_dir)) + ".torrent"
    if not path.exists(path.dirname(torrent)):
        os.makedirs(path.dirname(torrent))
    tracker_url = f"{tracker}{passkey}/announce"
    if source is None:
        command = ["mktorrent", "-p", "-a", tracker_url, "-o", torrent, input_dir]
    else:
        command = ["mktorrent", "-p", "-s", source, "-a", tracker_url, "-o", torrent, input_dir]
    subprocess.check_output(command, stderr=subprocess.STDOUT)
    return torrent


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("output_format", choices=encoders.keys())
    parser.add_argument(
        "-j", "--threads", default=multiprocessing.cpu_count(), type=int
    )
    args = parser.parse_args()

    transcode_release(
        os.path.expanduser(args.input_dir),
        os.path.expanduser(args.output_dir),
        args.output_format,
        args.threads,
    )


if __name__ == "__main__":
    main()
