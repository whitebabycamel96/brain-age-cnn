#!/bin/bash
# sync_to_pvc.sh
# Copies data files to PVC. Code comes from GitHub automatically.
# Run from your project root.

set -e

NAMESPACE="hdsi-schwartzman"
POD_NAME="vbm-transfer"

echo "============================================"
echo "  VBM AE — data sync to PVC"
echo "============================================"

POD_STATUS=$(kubectl get pod $POD_NAME -n $NAMESPACE \
    --no-headers -o custom-columns=":status.phase" 2>/dev/null || echo "NotFound")

if [ "$POD_STATUS" = "NotFound" ] || [ "$POD_STATUS" = "Succeeded" ] || [ "$POD_STATUS" = "Failed" ]; then
    echo "→ starting transfer pod ..."
    kubectl delete pod $POD_NAME -n $NAMESPACE --ignore-not-found=true
    sleep 2
    kubectl apply -f transfer_pod.yaml -n $NAMESPACE
    echo "→ waiting for pod to be ready ..."
    kubectl wait --for=condition=Ready pod/$POD_NAME -n $NAMESPACE --timeout=120s
elif [ "$POD_STATUS" = "Running" ]; then
    echo "→ transfer pod already running"
else
    kubectl wait --for=condition=Ready pod/$POD_NAME -n $NAMESPACE --timeout=120s
fi

# participants TSV
echo ""
echo "→ copying participants_study_3.tsv ..."
kubectl cp ./data/participants_study_3.tsv \
    $POD_NAME:/data/participants_study_3.tsv -n $NAMESPACE
echo "   done ✓"

# .npy files — copied individually directly into /data/vbm_data_sliced/
echo ""
echo "→ copying .npy files ..."
for npy in ./data/vbm_data_sliced/*.npy; do
    if [ -f "$npy" ]; then
        fname=$(basename "$npy")
        size=$(du -sh "$npy" | cut -f1)
        echo "   $fname ($size) ..."
        kubectl cp "$npy" \
            $POD_NAME:/data/vbm_data_sliced/"$fname" -n $NAMESPACE
        echo "   $fname ✓"
    fi
done

# verify
echo ""
echo "→ PVC contents:"
kubectl exec $POD_NAME -n $NAMESPACE -- find /data -type f | sort

echo ""
echo "============================================"
echo "  sync complete"
echo "  code comes from GitHub on pod startup"
echo "  to clean up: kubectl delete pod $POD_NAME -n $NAMESPACE"
echo "============================================"