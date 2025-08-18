#!/usr/bin/env python3

from sys import argv, exit
import json
import os


def main():
    torrent_hash = argv[5].upper()
    cache_file = os.path.expanduser("~/.orpheusmorebetter/cache-crawl")

    # find the hash and set done = true
    with open(cache_file, "r") as f:
        cache = json.load(f)

    for torrent in cache:
        if torrent["hash"] == torrent_hash:
            torrent["done"] = True
            with open(cache_file, "w") as f:
                json.dump(cache, f)
            exit(0)

    exit(1)
