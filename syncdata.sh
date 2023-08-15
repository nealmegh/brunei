#!/bin/bash
# Prompt the user for the source and destination directories
echo "Enter the source directory: "
read source
echo "Enter the destination directory: "
read destination

# Sync the directories interactively
rsync -avzi --progress $source $destination