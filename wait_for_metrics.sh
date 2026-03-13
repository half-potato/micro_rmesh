#!/bin/bash
while ! grep -q "^test_PSNR" run.log; do
  sleep 30
done
grep -E "^test_PSNR:|^n_vertices:" run.log
