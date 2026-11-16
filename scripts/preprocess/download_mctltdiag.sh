#!/bin/bash
cd "${WORK_DIR:?set WORK_DIR}/mctlt"
while read f n; do
  id=${n%.tar}
  [ -s tars/$n ] && tar tf tars/$n >/dev/null 2>&1 && continue
  for try in 1 2 3; do
    curl -sfL --retry 3 https://dataverse.harvard.edu/api/access/datafile/$f -o tars/$n.part && mv tars/$n.part tars/$n && break
    sleep 10
  done
  [ -s tars/$n ] || echo "FAIL $n"
done < tar_ids.txt
echo DONE
