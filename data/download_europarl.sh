#!/usr/bin/env bash

scriptdir=$(realpath $(dirname $0))
europarldatauri="https://www.statmt.org/europarl/v7/europarl.tgz"

echo $scriptdir

if [[ ! -d txt ]]; then
  wget https://www.statmt.org/europarl/v7/europarl.tgz
  tar xzf europarl.tgz
  mkdir -p txt_noxml
fi


