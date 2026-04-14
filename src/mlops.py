from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MlflowLogger:
    enabled: bool
    run: Any | None = None
    _mlflow: Any | None = None

    @classmethod
    def create(
        cls,
        enabled: bool,
        experiment_name: str = "voxelhouse-vae",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> "MlflowLogger":
        if not enabled:
            return cls(enabled=False)
        try:
            import mlflow  # type: ignore
        except ImportError:
            print("[WARN] mlflow is not installed; disabling mlflow logging")
            return cls(enabled=False)

        def _start_with_uri(uri: str | None) -> tuple[Any, str | None]:
            if uri:
                mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)
            run = mlflow.start_run(run_name=run_name)
            if tags:
                mlflow.set_tags(tags)
            return run, uri

        try:
            run, effective_uri = _start_with_uri(tracking_uri)
        except Exception as exc:
            fallback_uri = (Path.cwd() / "mlruns").resolve().as_uri()
            can_retry_with_fallback = not tracking_uri
            if not can_retry_with_fallback:
                print(f"[WARN] mlflow initialization failed ({exc}); disabling mlflow logging")
                return cls(enabled=False)
            try:
                print(
                    f"[WARN] mlflow initialization failed ({exc}); "
                    f"retrying with local tracking uri: {fallback_uri}"
                )
                run, effective_uri = _start_with_uri(fallback_uri)
            except Exception as fallback_exc:
                print(
                    f"[WARN] mlflow fallback initialization failed ({fallback_exc}); "
                    "disabling mlflow logging"
                )
                return cls(enabled=False)
        if effective_uri:
            print(f"[INFO] mlflow tracking uri: {effective_uri}")
        return cls(enabled=True, run=run, _mlflow=mlflow)

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled or self._mlflow is None:
            return
        clean = {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))}
        if clean:
            self._mlflow.log_params(clean)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled or self._mlflow is None:
            return
        if step is None:
            self._mlflow.log_metrics(metrics)
        else:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        if not self.enabled or self._mlflow is None:
            return
        self._mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_artifacts(self, path: str, artifact_path: str | None = None) -> None:
        if not self.enabled or self._mlflow is None:
            return
        self._mlflow.log_artifacts(path, artifact_path=artifact_path)

    def close(self) -> None:
        if self.enabled and self._mlflow is not None:
            self._mlflow.end_run()
            self.enabled = False

