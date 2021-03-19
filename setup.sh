#!/bin/bash

echo "Downloading STRING DB info..."
mkdir stringsdb-v11.0.nosync/
curl https://stringdb-static.org/download/protein.links.full.v11.0/9606.protein.links.full.v11.0.txt.gz > stringsdb-v11.0.nosync/9606.protein.links.full.v11.0.txt.gz
curl https://stringdb-static.org/download/protein.info.v11.0/9606.protein.info.v11.0.txt.gz > stringsdb-v11.0.nosync/9606.protein.info.v11.0.txt.gz
curl https://stringdb-static.org/download/protein.aliases.v11.0/9606.protein.aliases.v11.0.txt.gz > stringsdb-v11.0.nosync/9606.protein.aliases.v11.0.txt.gz
gunzip stringsdb-v11.0.nosync/*

echo "Note: The STRING file headers are occasionally not formatted properly. If there are any octothorpes ('#' symbols) on the first line, then please remove them."
echo "Note: You will also need to download the PDBbind refined and general sets at http://pdbbind.org.cn/download.php"
