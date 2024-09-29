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

rm src/default_paths.txt
echo "$this_path" >> src/default_paths.txt
echo "" >> src/default_paths.txt
echo "" >> src/default_paths.txt
