# Kubernetes & NRP Nautilus — A Working Reference

A practical guide to what Kubernetes is, how the NRP Nautilus setup works, what
the config file does, and how to actually use the cluster. Written around the
setup completed on macOS with the `hdsi-schwartzman` namespace.

---

## 1. What Kubernetes is

Kubernetes (often abbreviated **k8s** — "k", eight letters, "s") is a system for
running programs on a pool of many machines instead of one. You describe *what*
you want running, and Kubernetes decides *where* it runs, restarts it if it
dies, and manages its share of CPU, memory, and GPUs.

The key mental model: you don't log into a specific machine and start a process
the way you would over SSH. Instead, you submit a description of a workload to
the cluster's control plane, and the cluster places it on whichever node has
room. On NRP Nautilus, those nodes are spread across many universities (SDSU,
Georgia Tech, Fresno State, UCSD-linked nodes, and more), so "the cluster" is a
federated pool rather than a single data center.

### Core objects

| Object | What it is |
|---|---|
| **Node** | A physical or virtual machine in the cluster. You don't pick one; the scheduler does. |
| **Pod** | The smallest unit you run — one or more containers that share a network address and storage. Usually "a pod" = "one running container" in practice. |
| **Job** | A pod that runs to completion (e.g. a training run) and then stops. The right object for batch work. |
| **Deployment** | A pod kept running indefinitely and restarted if it dies (e.g. a web service). Not what you want for training. |
| **PVC (PersistentVolumeClaim)** | A request for durable storage that survives pod restarts. This is where your data must live. |
| **Namespace** | A named slice of the cluster you have access to. Yours is `hdsi-schwartzman`. You only see resources inside it. |

### The single most important property: pods are stateless and interruptible

A pod can be killed and restarted at any time, with no warning. **Anything
written to the pod's local filesystem is gone forever on restart.** Restarts are
normal and expected — not a failure condition.

This is the biggest difference from a SLURM cluster like SDSC Expanse, where a
job holds an allocation on specific nodes and the filesystem persists across the
job's life. On Kubernetes you must assume your container can vanish mid-run, so:

- Persistent data (datasets, caches, checkpoints, outputs) must live on a **PVC**, never on the pod's local disk.
- Long-running work must **checkpoint frequently and resume cleanly** from the latest checkpoint, because interruption can come mid-epoch.

Two hard rules on Nautilus specifically:

- **Never** run a `Job` with `sleep` or any command that never ends on its own — that is a bannable offense.
- **Never** force-delete pods.

---

## 2. How the access setup works

You authenticate to the cluster with `kubectl`, the Kubernetes command-line
tool. `kubectl` reads a **config file** that tells it (a) where the cluster's
API server is and (b) how to prove who you are.

On NRP, identity is handled by **OpenID Connect (OIDC)** through **CILogon** —
the same institutional login you use on the web portal. `kubectl` can't speak
OIDC on its own, so a plugin called **kubelogin** handles the browser login flow
and hands the resulting token back to `kubectl`.

The chain looks like this:

```
kubectl  →  reads ~/.kube/config  →  needs a token
         →  calls kubelogin (kubectl-oidc_login plugin)
         →  opens browser → CILogon institutional login
         →  token returned to kubectl, cached ~30 min
         →  kubectl talks to the cluster API server
```

You set this up once. After that, the token refreshes automatically.

### Setup steps (what was done)

1. **Install `kubectl`** — the Kubernetes CLI.

2. **Install the `kubelogin` plugin.** This is mandatory; the config will not
   work without it. Easiest install on macOS:
   ```bash
   brew install int128/kubelogin/kubelogin
   ```
   The critical requirement: the binary must be discoverable on your `PATH` as
   `kubectl-oidc_login` (with an underscore). Homebrew / Krew / Chocolatey
   handle this naming for you; a manual GitHub-release install does not.

3. **Download the cluster config** and save it to `~/.kube/config`:
   ```bash
   mkdir -p ~/.kube
   curl -o ~/.kube/config -fSL "https://nrp.ai/config"
   ```
   This file already contains NRP's API server address, the OIDC issuer URL, and
   the client ID — so there is **no** separate "set up OIDC provider" step. That
   part of the upstream kubelogin README is for people building their own
   cluster, not for users of a managed one like Nautilus.

4. **Select the context:**
   ```bash
   kubectl config use-context nautilus
   ```

5. **Set your default namespace** so you don't pass `-n` on every command:
   ```bash
   kubectl config set contexts.nautilus.namespace hdsi-schwartzman
   ```

6. **Trigger the login** by running any real command:
   ```bash
   kubectl get nodes
   ```
   A CILogon browser window opens, you authenticate with your institution, the
   window closes, and a node list comes back. That browser step **is** "logging
   in to the OIDC provider" — there's no separate command for it.

---

## 3. What the config file is

The config file (`~/.kube/config`) is a YAML file with three kinds of entries:

- **clusters** — where the API server lives (its address and TLS certificate).
  For NRP this points at the Nautilus API server.
- **users** — how to authenticate. The NRP file uses an `exec` block that calls
  `kubelogin` (`oidc-login get-token`) with the issuer URL and client ID
  pre-filled. This is why you never typed those values yourself.
- **contexts** — a named pairing of (cluster + user + default namespace). NRP's
  is called `nautilus`; you added `hdsi-schwartzman` as its default namespace.

A trimmed view of the user section looks roughly like:

```yaml
users:
  - name: oidc
    user:
      exec:
        apiVersion: client.authentication.k8s.io/v1
        command: kubectl
        args:
          - oidc-login
          - get-token
          - --oidc-issuer-url=...      # pre-filled by NRP
          - --oidc-client-id=...       # pre-filled by NRP
```

To inspect your own config at any time:

```bash
kubectl config view              # full config (secrets redacted)
kubectl config get-contexts      # list contexts; * marks the active one
kubectl config current-context   # just the active context name
```

### Common config problems and what they mean

| Symptom | Cause | Fix |
|---|---|---|
| `connection to localhost:8080 refused` | No usable config found — kubectl fell back to its default. | Check the file exists at `~/.kube/config`; re-download if missing. |
| Config saved as `config.txt` / `config.yaml` | Browser added an extension. | `mv ~/.kube/config.txt ~/.kube/config` |
| `head` of the file shows `<!DOCTYPE html>` | Download grabbed an error page, not the config. | Re-run the `curl`; make sure you're logged into the portal in your browser. |
| `Forbidden` on your namespace | Membership not yet in your token. | `kubectl oidc-login clean`, then retry. |
| One `context deadline exceeded` line, then results appear | First request timed out during auth handshake; harmless. | Ignore — kubectl retried and succeeded. |

---

## 4. How to use the cluster

### Everyday commands

```bash
# See what's running in your namespace
kubectl get pods

# Detailed status of one pod
kubectl describe pod <pod-name>

# Stream logs from a pod
kubectl logs -f <pod-name>

# Open a shell inside a running pod
kubectl exec -it <pod-name> -- /bin/bash

# Submit something to the cluster
kubectl apply -f my-job.yaml

# Remove it
kubectl delete -f my-job.yaml

# Cluster-wide node list (works; everyone can see nodes)
kubectl get nodes
```

`No resources found in hdsi-schwartzman namespace.` is the normal, healthy
response to `kubectl get pods` when nothing is running yet — it confirms access,
it is not an error.

### Refreshing membership / tokens

The token caches for about 30 minutes and refreshes automatically. If you were
just added to a new namespace and it isn't visible yet, force a fresh token:

```bash
kubectl oidc-login clean
kubectl get nodes        # triggers a new login
```

### Running a workload

You describe a workload in a YAML manifest and submit it with `kubectl apply -f`.
A GPU batch job declares, at minimum: the container image, the command to run,
the GPU/CPU/memory it requests, and any PVC it mounts for persistent storage.

The disciplines that matter most for long training runs on this cluster:

1. **Mount a PVC** for anything that must survive — datasets, caches,
   checkpoints, outputs. Never write important data to the pod's local disk.
2. **Request GPUs explicitly** in the pod spec (e.g. `nvidia.com/gpu: 2`), along
   with matching CPU and memory requests.
3. **Checkpoint often and resume from the latest checkpoint**, because the pod
   can be interrupted mid-run without warning. A resume mechanism that already
   works across separate jobs is the right foundation; the new requirement
   versus SLURM is tolerating an *unannounced* mid-epoch kill.

### GUI options

If you'd rather not live in the terminal, these all read the same
`~/.kube/config`:

- **K9s** — terminal UI, fast for watching pods and logs.
- **Lens** — full graphical desktop app.
- **VS Code** — with the Kubernetes and Remote Development extensions. Note: the
  config must be exactly at `~/.kube/config`, and both `kubectl` and
  `kubectl-oidc_login` must be on your `PATH` for it to authenticate.

---

## 5. Quick reference card

```bash
# --- one-time setup ---
brew install int128/kubelogin/kubelogin
mkdir -p ~/.kube
curl -o ~/.kube/config -fSL "https://nrp.ai/config"
kubectl config use-context nautilus
kubectl config set contexts.nautilus.namespace hdsi-schwartzman

# --- verify ---
kubectl get nodes        # triggers CILogon browser login
kubectl get pods         # "No resources found" = success

# --- daily use ---
kubectl get pods
kubectl logs -f <pod>
kubectl exec -it <pod> -- /bin/bash
kubectl apply -f job.yaml
kubectl delete -f job.yaml

# --- when membership/token is stale ---
kubectl oidc-login clean
```

**Remember:** containers are stateless. Persist everything important to a PVC,
checkpoint often, and expect restarts.

---

## Useful links

- Getting Started: https://nrp.ai/documentation/userdocs/start/getting-started
- Using Nautilus: https://nrp.ai/documentation/userdocs/start/using-nautilus
- Cluster Policies: https://nrp.ai/documentation/userdocs/start/policies
- GPU pods: https://nrp.ai/documentation/userdocs/running/gpu-pods
- Storage (Ceph): https://nrp.ai/documentation/userdocs/storage/ceph
- Namespaces manager: https://nrp.ai/namespaces
- kubectl cheatsheet: https://kubernetes.io/docs/reference/kubectl/cheatsheet/