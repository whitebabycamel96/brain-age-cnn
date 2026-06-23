#!/bin/bash
# sync_to_pvc.sh
#
# Copies data files from your local machine onto the PVC.
# Code is handled automatically via GitHub on pod startup —
# this script is only needed for data files (.npy, .tsv).
#
# Usage:
#   chmod +x sync_to_pvc.sh
#   ./sync_to_pvc.sh
#
# Run from your project root (same directory as vbm_data_sliced/).

set -e

NAMESPACE="hdsi-schwartzman"
POD_NAME="vbm-transfer"

echo "============================================"
echo "  VBM AE — data sync to PVC"
echo "  namespace: $NAMESPACE"
echo "============================================"

# ── ensure transfer pod is running ───────────────────────────────────
POD_STATUS=$(kubectl get pod $POD_NAME -n $NAMESPACE \
    --no-headers -o custom-columns=":status.phase" 2>/dev/null || echo "NotFound")

if [ "$POD_STATUS" = "NotFound" ] || [ "$POD_STATUS" = "Succeeded" ] || [ "$POD_STATUS" = "Failed" ]; then
    echo "→ starting transfer pod ..."
    kubectl delete pod $POD_NAME -n $NAMESPACE --ignore-not-found=true
    sleep 2
    kubectl apply -f transfer_pod.yaml -n $NAMESPACE
    echo "→ waiting for pod to be ready ..."
    kubectl wait --for=condition=Ready pod/$POD_NAME \
        -n $NAMESPACE --timeout=120s
elif [ "$POD_STATUS" = "Running" ]; then
    echo "→ transfer pod already running"
else
    echo "→ waiting for pod ..."
    kubectl wait --for=condition=Ready pod/$POD_NAME \
        -n $NAMESPACE --timeout=120s
fi

# ── copy participants TSV ─────────────────────────────────────────────
echo ""
echo "→ copying participants_study_3.tsv ..."
kubectl cp participants_study_3.tsv \
    $POD_NAME:/data/participants_study_3.tsv -n $NAMESPACE
echo "   done ✓"

# ── copy .npy files directly (no nesting) ────────────────────────────
echo ""
echo "→ copying .npy files ..."
for npy in vbm_data_sliced/*.npy; do
    if [ -f "$npy" ]; then
        fname=$(basename "$npy")
        size=$(du -sh "$npy" | cut -f1)
        echo "   $fname ($size) ..."
        kubectl cp "$npy" \
            $POD_NAME:/data/vbm_data_sliced/"$fname" -n $NAMESPACE
        echo "   $fname ✓"
    fi
done

# ── verify ────────────────────────────────────────────────────────────
echo ""
echo "→ verifying /data on PVC ..."
kubectl exec $POD_NAME -n $NAMESPACE -- find /data -type f | sort

echo ""
echo "============================================"
echo "  sync complete"
echo "  to clean up: kubectl delete pod $POD_NAME -n $NAMESPACE"
echo "============================================"