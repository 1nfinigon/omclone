#!/usr/bin/env bash

set -e -u -x -o pipefail

pushd ..
[ ! -d om-leaderboard ] && git clone https://github.com/F43nd1r/om-leaderboard.git
[ ! -d omsim ] && git clone https://github.com/ianh/omsim.git
popd

mkdir -p test
pushd test
this_path="$(pwd)"
ln -sf ../../omsim/test/puzzle .
ln -sf ../../omsim/test/solution .
ln -sf ../../om-leaderboard ./om-leaderboard-master
popd

if [ ! -f src/default_paths.txt ]; then
    echo "$this_path" >> src/default_paths.txt
    echo "" >> src/default_paths.txt
    echo "" >> src/default_paths.txt
fi

mkdir test/net
mkdir test/games
mkdir test/tensorboard
mkdir test/training_data
