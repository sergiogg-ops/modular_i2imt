#!/bin/bash

# Check if a directory is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <config_directory>"
    exit 1
fi

CONFIG_DIR="$1"

# Check if the directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Directory '$CONFIG_DIR' not found."
    exit 1
fi

# Loop through all .yaml files in the directory
find "$CONFIG_DIR" -name "*.yaml" -print0 | while IFS= read -r -d $'\0' config_file; do
    echo "Processing config file: $config_file"
    echo "de-en"
    python data_generator.py --config "$config_file" --src_text wmt15/de-en/test.de --tgt_text wmt15/de-en/test.en \
    --src_lang de --tgt_lang en --output data/"$config_file"/de-en/test
    echo "en-fr"
    python data_generator.py --config "$config_file" --src_text wmt15/fr-en/test.fr --tgt_text wmt15/fr-en/test.en \
    --src_lang fr --tgt_lang en --output data/"$config_file"/fr-en/test
done

echo "Finished processing all config files."