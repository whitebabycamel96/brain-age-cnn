# push changes to the branch
git add . && git commit -m "your message" && git push origin autoenocder_age

# resubmit the pod
kubectl delete pod vbm-train --ignore-not-found=true
kubectl apply -f train_job_github.yaml
kubectl logs -f vbm-train


chmod +x sync_to_pvc.sh
./sync_to_pvc.sh