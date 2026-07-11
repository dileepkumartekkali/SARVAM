# infra/

- **docker** — the backend image is defined at [`backend/Dockerfile`](../backend/Dockerfile)
  (kept next to its build context); `docker-compose.yml` at the repo root wires
  it with redis + postgres for local dev. No separate `infra/docker` dir — it
  would only duplicate that.
- **k8s/** — Kubernetes manifest **stubs**. Not production-ready: no resource
  limits, no HPA, no connection-draining rollout config. Voice sessions are
  long-lived stateful WebSockets — scale on concurrent connections and use
  connection-draining rollouts (S2S plan §7) before these go real.
- **terraform/** — provider/module **stubs** only. No backend state config, no
  real resources.
