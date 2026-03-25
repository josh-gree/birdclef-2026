_poll kernel:
    #!/usr/bin/env bash
    set -e
    echo "Waiting for {{kernel}}..."
    while true; do
        status=$(kaggle kernels status {{kernel}} 2>&1)
        echo "  $status"
        if echo "$status" | grep -q "COMPLETE"; then
            echo "Done."
            break
        elif echo "$status" | grep -q "ERROR\|CANCEL"; then
            echo "Failed!"
            exit 1
        fi
        sleep 15
    done

build-deps:
    kaggle kernels push -p notebooks/deps
    just _poll joshgreen/birdclef-2026-deps

build-lib:
    kaggle kernels push -p notebooks/lib
    just _poll joshgreen/birdclef-2026-lib

build: build-deps build-lib

submit checkpoint:
    #!/usr/bin/env bash
    set -e
    tmp=$(mktemp -d)
    echo "Downloading checkpoint from Modal..."
    modal volume get birdclef-2026-storage {{checkpoint}} "$tmp/checkpoint.pt"
    echo "Uploading to Kaggle dataset..."
    cat > "$tmp/dataset-metadata.json" <<EOF
    {"id": "joshgreen/birdclef-2026-checkpoint", "licenses": [{"name": "CC0-1.0"}]}
    EOF
    kaggle datasets version -p "$tmp" -m "{{checkpoint}}"
    rm -rf "$tmp"
    echo "Running submission notebook..."
    kaggle kernels push -p notebooks
    just _poll joshgreen/birdclef-2026-submission
