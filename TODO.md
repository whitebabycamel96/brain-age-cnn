# push changes to the branch
git add . && git commit -m "your message" && git push origin autoenocder_age

# resubmit the pod
kubectl delete pod vbm-train --ignore-not-found=true
kubectl apply -f train_job_github.yaml
kubectl logs -f vbm-train


chmod +x sync_to_pvc.sh
./sync_to_pvc.sh




/data/
  train.py              ← from GitHub
  model.py              ← from GitHub
  config.json           ← from GitHub
  VBMAgeDataset.py      ← from GitHub
  preprocessing.py      ← from GitHub
  hparam_search.py      ← from GitHub
  participants_study_3.tsv   ← from sync script
  vbm_data_sliced/
    vbm_z60.npy         ← from sync script
    vbm_z40.npy
    vbm_y80.npy
  checkpoints/
    autoencoder_age/    ← written by training job
  hparam_search/        ← written by hparam job