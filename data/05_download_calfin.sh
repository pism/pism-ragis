
#!/bin/bash

mkdir -p calfin
wget -nc https://datadryad.org/stash/downloads/file_stream/458630
mv 458630 calfin/
cd calfin
unzip 458630
rm 458630
cd ..
