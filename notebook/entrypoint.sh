#!/bin/bash

set -euo pipefail

CLOUD_PROVIDER=${CLOUD_PROVIDER:=}

# Run startup scripts
if [ -d /etc/startup.d ]; then
  for i in /etc/startup.d/*.sh; do
    if [ -r $i ]; then
      source $i
    fi
  done
  unset i
fi
wait=0

# Wait for bucket to be mounted when cloud provider is not local.
# When bucket is mounted, the ".mounted" file gets written by the
# mounting script.
if [[ -n "${CLOUD_PROVIDER}" ]] && [[ "${CLOUD_PROVIDER}" != "local" ]]; then
  while [ ! -f "${DATA_DIR}/.mounted" ]; do
    sleep 1;
    wait=$((wait+1));
    if [ $wait -eq 60 ]; then
      echo "timed out after waiting 60 seconds for bucket to mount"
      exit 1
    fi;
  done;
fi
exec "$@"

