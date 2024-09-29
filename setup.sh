#!/usr/bin/env bash

set -e -u -x -o pipefail

pushd ..
git clone https://github.com/F43nd1r/om-leaderboard.git
git clone https://github.com/ianh/omsim.git
popd

mkdir -p test
pushd test
this_path="$(pwd)"
ln -s ../omsim/test/puzzle .
ln -s ../omsim/test/solution .
ln -s ../om-leaderboard ./om-leaderboard-master
popd

echo "$this_path" >> src/default_paths.txt
echo "" >> src/default_paths.txt
echo "" >> src/default_paths.txt
