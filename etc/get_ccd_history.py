#!/usr/bin/env python3
"""
Script to extract CCD change history from DESI spectrograph YAML files.
Returns a dictionary of CCD changes with start and end dates.

Usage:

    # Print CCD changes to screen
    ./get_ccd_history.py

    # Save to YAML file
    ./get_ccd_history.py -o ccd_history.yaml

    # Use custom path (replaces $DESI_SPECTRO_CALIB/spec)
    ./get_ccd_history.py -p /path/to/spec -o ccd_history.yaml

"""

import yaml
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict


def parse_date(date_str):
    """Convert date string (YYYYMMDD) to datetime object."""
    date_str = str(date_str).strip().strip("'\"")
    try:
        return datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        # Handle invalid dates by returning None
        return None


def get_previous_day(date_obj):
    """Get the day before the given date."""
    return date_obj - timedelta(days=1)


def get_ccd_history(base_path=None, output_path=None):
    """
    Parse YAML files and extract CCD changes.

    Parameters:
    -----------
    base_path : str, optional
        Base directory containing sm[1-10] subdirectories.
        Defaults to $DESI_SPECTRO_CALIB/spec if environment variable is set,
        otherwise uses current directory.
    output_path : str, optional
        If provided, save the CCD changes dictionary to this path in YAML format.

    Returns:
    --------
    dict : Dictionary with structure:
        {
            'sm1-b': [
                {
                    'sn22822': {
                        'date_obs_begin': '20221108',
                        'date_obs_end': None  # None for current/most recent
                    }
                },
                {
                    'sn17986': {
                        'date_obs_begin': '20191106',
                        'date_obs_end': '20221107'  # day before next change
                    }
                },
                ...
            ],
            ...
        }
    """
    if base_path is None:
        # Use environment variable if set, otherwise current directory
        desi_calib = os.environ.get('DESI_SPECTRO_CALIB')
        if desi_calib:
            base_path = os.path.join(desi_calib, 'spec')
        else:
            base_path = '.'
    print('DESI_SPECTRO_CALIB = {}'.format(base_path))

    base_path = Path(base_path)
    ccd_history = {}

    # Find all YAML files matching pattern sm[1-10]/sm[1-10]-[b,r,z].yaml
    # Sort naturally (sm1, sm2, ..., sm10) instead of lexicographically
    import re
    def natural_sort_key(path):
        parts = re.split(r'(\d+)', str(path.stem))
        return [int(p) if p.isdigit() else p for p in parts]

    yaml_files = sorted(base_path.glob('sm*/sm*-*.yaml'), key=natural_sort_key)

    for yaml_file in yaml_files:
        # Extract camera name (e.g., 'sm1-b')
        camera_name = yaml_file.stem

        # Read YAML file
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        if not data:
            continue

        # Get the top-level key (should be the camera name)
        top_key = list(data.keys())[0]
        versions = data[top_key]

        # Extract CCD and date information for each version
        detector_entries = []
        for version_name, version_data in versions.items():
            if 'DETECTOR' in version_data and 'DATE-OBS-BEGIN' in version_data:
                detector = version_data['DETECTOR']
                date_begin = str(version_data['DATE-OBS-BEGIN']).strip().strip("'\"")

                # Skip entries with invalid dates
                if parse_date(date_begin) is not None:
                    detector_entries.append({
                        'detector': detector,
                        'date_obs_begin': date_begin,
                        'version': version_name
                    })

        # Sort by date (oldest first)
        detector_entries.sort(key=lambda x: parse_date(x['date_obs_begin']))

        # Handle conflicts: when multiple versions have same DATE-OBS-BEGIN,
        # keep only the one with the most recent version name
        filtered_entries = []
        i = 0
        while i < len(detector_entries):
            current_date = detector_entries[i]['date_obs_begin']

            # Collect all entries with the same date
            same_date_entries = []
            while i < len(detector_entries) and detector_entries[i]['date_obs_begin'] == current_date:
                same_date_entries.append(detector_entries[i])
                i += 1

            # If multiple entries with same date, keep the one with most recent version name
            if len(same_date_entries) > 1:
                # Sort by version name (descending) to get most recent
                same_date_entries.sort(key=lambda x: x['version'], reverse=True)
                filtered_entries.append(same_date_entries[0])
            else:
                filtered_entries.append(same_date_entries[0])

        detector_entries = filtered_entries

        # Identify CCD changes
        changes = []
        prev_detector = None

        for i, entry in enumerate(detector_entries):
            current_detector = entry['detector']

            # Add entry if CCD changed or it's the first entry
            if current_detector != prev_detector:
                # Calculate end date: day before next different detector
                date_obs_end = None
                for j in range(i + 1, len(detector_entries)):
                    if detector_entries[j]['detector'] != current_detector:
                        next_date = parse_date(detector_entries[j]['date_obs_begin'])
                        date_obs_end = get_previous_day(next_date).strftime('%Y%m%d')
                        break

                # Only add valid entries where end_date is after start_date (or None)
                if date_obs_end is None or parse_date(date_obs_end) >= parse_date(entry['date_obs_begin']):
                    changes.append({
                        current_detector: {
                            'date_obs_begin': entry['date_obs_begin'],
                            'date_obs_end': date_obs_end
                        }
                    })

                prev_detector = current_detector

        # Merge consecutive periods with the same detector
        merged_changes = []
        for change in changes:
            # Get CCD name (the only key in the dict)
            detector = list(change.keys())[0]

            # Check if last entry has same detector
            if merged_changes:
                last_detector = list(merged_changes[-1].keys())[0]
                if last_detector == detector:
                    # Extend the end date of the previous period
                    merged_changes[-1][detector]['date_obs_end'] = change[detector]['date_obs_end']
                else:
                    merged_changes.append(change)
            else:
                merged_changes.append(change)

        if merged_changes:
            ccd_history[camera_name] = merged_changes

    # Save to file if output_path is provided
    if output_path:
        with open(output_path, 'w') as f:
            yaml.dump(ccd_history, f, default_flow_style=False, sort_keys=False)

    return ccd_history


def print_ccd_changes(ccd_history):
    """Pretty print CCD changes."""
    # Dictionary is already in natural order from get_ccd_history()

    # Spectrograph SM <-> SP
    sp2sm = {0: 4, 1: 10, 2: 5, 3: 6, 4: 1, 5: 9, 6: 7, 7: 8, 8: 2, 9: 3}
    sm2sp = {aa: bb for bb, aa in sp2sm.items()}

    for camera, changes in ccd_history.items():
        petal = sm2sp[int(camera.split('-')[0].replace('sm', ''))]
        print(f"\n{camera}{petal}:")
        for i, change in enumerate(changes, 1):
            detector = list(change.keys())[0]
            info = change[detector]
            end_str = info['date_obs_end'] if info['date_obs_end'] else 'present'
            print(f"  {i}. {detector}: {info['date_obs_begin']} to {end_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract detector changes from DESI spectrograph YAML files.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-p', '--path',
        dest='base_path',
        help='Base directory containing sm[1-10] subdirectories. '
             'Default: $DESI_SPECTRO_CALIB/spec if set, otherwise current directory.'
    )

    parser.add_argument(
        '-o', '--output',
        dest='output_path',
        help='Output file path to save detector changes in YAML format. '
             'If not specified, results are printed to stdout.'
    )

    args = parser.parse_args()

    # Get CCD changes
    changes = get_ccd_history(base_path=args.base_path, output_path=args.output_path)

    # Print results (unless saving to file only)
    if not args.output_path:
        print("CCD history in DESI Spectrograph")
        print("=" * 50)
        print_ccd_changes(changes)
    else:
        print(f"CCD history saved to {args.output_path}")
