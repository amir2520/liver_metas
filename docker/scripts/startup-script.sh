#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

if [[ "${IS_PROD_ENV}" == 'true' ]]; then
    echo "Production mode: Starting SSH tunnel to GCP MLflow server..."
    echo "Connecting to ${VM_NAME} in zone ${ZONE}..."
    
    # Bind to 0.0.0.0 instead of localhost
    /usr/local/gcloud/google-cloud-sdk/bin/gcloud compute ssh "${VM_NAME}" \
        --zone "${ZONE}" \
        --tunnel-through-iap \
        -- -N -L "0.0.0.0:${PROD_MLFLOW_SERVER_PORT}:localhost:${PROD_MLFLOW_SERVER_PORT}"
else
    echo "Development mode: Starting local MLflow tracking server..."
    /start-tracking-server.sh &
    tail -f /dev/null
fi