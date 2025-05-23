#!/bin/bash

# valid dataset
VALID_DATASETS="eurlex-4k wiki10-31k amazon-3m amazon-670k wiki-500k amazoncat-13k"
DOWNLOAD_BASE_URL="https://archive.org/download/pecos-dataset/xmc-base"
DATASET_DIR="./xmc-base"

# create dir
mkdir -p "$DATASET_DIR"

# pares argument
DATASET=${1:-"eurlex-4k"}  # default dataset

# validate dataset
if ! echo "$VALID_DATASETS" | grep -w "$DATASET" > /dev/null; then
    echo "error: novalid name '$DATASET'"
    echo "option datset: $VALID_DATASETS"
    exit 1
fi

echo "select data: ${DATASET}"

# check if dataset exists
if [ -d "${DATASET_DIR}/${DATASET}" ]; then
    echo "dataset: ${DATASET} has existen in ${DATASET_DIR} "
    exit 0
fi

# download dataset
echo "downloading ${DATASET}..."
if wget -nc "${DOWNLOAD_BASE_URL}/${DATASET}.tar.gz" -P "$DATASET_DIR"; then
    echo "uncompressing..."
    
    if tar -xzf "${DATASET_DIR}/${DATASET}.tar.gz" -C "$DATASET_DIR"; then
        echo "finish..."
        
        # clean
        mv "${DATASET_DIR}/xmc-base/${DATASET}" "$DATASET_DIR/"
        rm -rf "${DATASET_DIR}/xmc-base"
        rm "${DATASET_DIR}/${DATASET}.tar.gz"
        
        echo "dataset ${DATASET} cleaned"
    else
        echo "error: uncompressing failed"
        rm -f "${DATASET_DIR}/${DATASET}.tar.gz"
        exit 1
    fi
else
    echo "error: downloading failed"
    exit 1
fi