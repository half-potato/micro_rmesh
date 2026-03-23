#!/bin/bash
while true; do
    if grep -q "test_PSNR" run.log 2>/dev/null; then
        grep "test_PSNR\|test_SSIM\|test_LPIPS\|n_vertices\|n_tets\|n_interior" run.log
        exit 0
    fi
    if grep -q "Traceback\|AssertionError\|AssertionError" run.log 2>/dev/null; then
        echo "ERROR detected"
        tail -10 run.log
        exit 1
    fi
    sleep 10
done
