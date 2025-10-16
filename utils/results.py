import math
from typing import List, Dict, Any, Tuple
import mlflow


def _mean_ci(values: List[float], z: float = 1.96) -> Tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(values) / n
    if n == 1:
        return (mean, mean, mean)
    # Sample standard deviation
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    return (mean, mean - z * se, mean + z * se)


def _extract_macro_metrics(report: Dict[str, Any]) -> Dict[str, float]:
    macro = report.get("macro avg", {})
    return {
        "precision": float(macro.get("precision", float("nan"))),
        "recall": float(macro.get("recall", float("nan"))),
        "f1": float(macro.get("f1-score", float("nan"))),
    }


def _aggregate_details(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Gather arrays
    accs: List[float] = []
    losses: List[float] = []
    macro_precisions: List[float] = []
    macro_recalls: List[float] = []
    macro_f1s: List[float] = []
    mean_aurocs: List[float] = []

    # Per-class AUROC aggregation structures
    per_class_aurocs: Dict[int, List[float]] = {}

    # Per-class PRF1 aggregation based on classification_report
    per_class_prf1: Dict[str, Dict[str, List[float]]] = {}

    for d in details:
        accs.append(float(d.get("validation_accuracy", float("nan"))))
        losses.append(float(d.get("validation_loss", float("nan"))))
        metrics = d.get("metrics", {})
        report = metrics.get("classification_report", {})
        macro = _extract_macro_metrics(report)
        macro_precisions.append(macro["precision"])
        macro_recalls.append(macro["recall"])
        macro_f1s.append(macro["f1"])

        auroc_list = metrics.get("auroc", []) or []
        if len(auroc_list) > 0:
            mean_aurocs.append(float(sum(auroc_list) / len(auroc_list)))
            # Populate per-class lists
            for idx, v in enumerate(auroc_list):
                per_class_aurocs.setdefault(idx, []).append(float(v))

        # Aggregate per-class PRF1
        for label, stats in report.items():
            if label in ("accuracy", "macro avg", "weighted avg"):
                continue
            if isinstance(stats, dict):
                per_class_prf1.setdefault(
                    label, {"precision": [], "recall": [], "f1": []}
                )
                per_class_prf1[label]["precision"].append(
                    float(stats.get("precision", float("nan")))
                )
                per_class_prf1[label]["recall"].append(
                    float(stats.get("recall", float("nan")))
                )
                per_class_prf1[label]["f1"].append(
                    float(stats.get("f1-score", float("nan")))
                )

    agg: Dict[str, Any] = {"n": len(details)}

    # Scalar aggregates with CI
    acc_mean, acc_lo, acc_hi = _mean_ci(accs)
    loss_mean, loss_lo, loss_hi = _mean_ci(losses)
    mp_mean, mp_lo, mp_hi = _mean_ci(macro_precisions)
    mr_mean, mr_lo, mr_hi = _mean_ci(macro_recalls)
    mf_mean, mf_lo, mf_hi = _mean_ci(macro_f1s)
    if len(mean_aurocs) > 0:
        au_mean, au_lo, au_hi = _mean_ci(mean_aurocs)
    else:
        au_mean = au_lo = au_hi = float("nan")

    agg["validation_accuracy"] = {"mean": acc_mean, "ci_low": acc_lo, "ci_high": acc_hi}
    agg["validation_loss"] = {"mean": loss_mean, "ci_low": loss_lo, "ci_high": loss_hi}
    agg["macro_avg"] = {
        "precision": {"mean": mp_mean, "ci_low": mp_lo, "ci_high": mp_hi},
        "recall": {"mean": mr_mean, "ci_low": mr_lo, "ci_high": mr_hi},
        "f1": {"mean": mf_mean, "ci_low": mf_lo, "ci_high": mf_hi},
    }
    agg["auroc_macro_mean"] = {"mean": au_mean, "ci_low": au_lo, "ci_high": au_hi}

    # Per-class AUROC aggregates
    if per_class_aurocs:
        agg["auroc_per_class"] = []
        # Sort by class index for stable order
        for idx in sorted(per_class_aurocs.keys()):
            m, lo, hi = _mean_ci(per_class_aurocs[idx])
            agg["auroc_per_class"].append(
                {"class_index": idx, "mean": m, "ci_low": lo, "ci_high": hi}
            )

    # Per-class PRF1 aggregates
    if per_class_prf1:
        agg["per_class"] = {}
        for label, vals in per_class_prf1.items():
            p_m, p_lo, p_hi = (
                _mean_ci(vals["precision"])
                if vals["precision"]
                else (float("nan"),) * 3
            )
            r_m, r_lo, r_hi = (
                _mean_ci(vals["recall"]) if vals["recall"] else (float("nan"),) * 3
            )
            f_m, f_lo, f_hi = (
                _mean_ci(vals["f1"]) if vals["f1"] else (float("nan"),) * 3
            )
            agg["per_class"][label] = {
                "precision": {"mean": p_m, "ci_low": p_lo, "ci_high": p_hi},
                "recall": {"mean": r_m, "ci_low": r_lo, "ci_high": r_hi},
                "f1": {"mean": f_m, "ci_low": f_lo, "ci_high": f_hi},
            }

    return agg


def save_results(new_results, experiment_name, cross_validation=False):
    """
    Save results to MLFlow.
    All results are logged as metrics and artifacts in MLFlow.
    """
    if cross_validation:
        # new_results: Dict[str, List[Dict]] mapping model_name -> list of fold results
        models_to_folds = new_results

        # Compute aggregates per model
        aggregated: Dict[str, Dict[str, Any]] = {}
        output: Dict[str, Any] = {}
        for model_name, folds in models_to_folds.items():
            agg = _aggregate_details(folds)
            aggregated[model_name] = agg
            output[model_name] = {
                "name": model_name,
                "folds": folds,
                "aggregate_metrics": agg,
            }

        # Log to MLFlow (create a summary run for the experiment)
        _log_cross_validation_to_mlflow(experiment_name, output, aggregated)
        return

    all_results = {"models": {}}

    # new_results: List[Dict] where each dict has key 'details' (list per split)
    for res in new_results:
        model_name = res["name"]
        details = res.get("details", [])
        agg = _aggregate_details(details) if details else {}
        res["aggregate_metrics"] = agg
        all_results["models"][model_name] = res

    # Log to MLFlow (create a summary run for the experiment)
    _log_train_validation_to_mlflow(experiment_name, all_results)


def _log_cross_validation_to_mlflow(experiment_name, output, aggregated):
    """Log cross-validation summary to MLFlow."""
    try:
        mlflow.set_experiment(f"{experiment_name}_summary")
        with mlflow.start_run(run_name=f"cross_validation_summary"):
            mlflow.set_tag("type", "cross_validation_summary")

            # Log aggregate metrics for each model
            for model_name, model_data in output.items():
                agg = model_data.get("aggregate_metrics", {})
                if agg:
                    # Log validation accuracy
                    val_acc = agg.get("validation_accuracy", {})
                    if val_acc:
                        mlflow.log_metric(
                            f"{model_name}_val_acc_mean", float(val_acc.get("mean", 0))
                        )
                        mlflow.log_metric(
                            f"{model_name}_val_acc_ci_low",
                            float(val_acc.get("ci_low", 0)),
                        )
                        mlflow.log_metric(
                            f"{model_name}_val_acc_ci_high",
                            float(val_acc.get("ci_high", 0)),
                        )

                    # Log validation loss
                    val_loss = agg.get("validation_loss", {})
                    if val_loss:
                        mlflow.log_metric(
                            f"{model_name}_val_loss_mean",
                            float(val_loss.get("mean", 0)),
                        )

                    # Log macro metrics
                    macro_avg = agg.get("macro_avg", {})
                    if macro_avg:
                        for metric_name, metric_vals in macro_avg.items():
                            if isinstance(metric_vals, dict):
                                mlflow.log_metric(
                                    f"{model_name}_macro_{metric_name}_mean",
                                    float(metric_vals.get("mean", 0)),
                                )

            # Log complete results as JSON artifact
            mlflow.log_dict(output, "cross_validation_results.json")

            # Log aggregated results as JSON artifact
            mlflow.log_dict(aggregated, "cross_validation_aggregated.json")

            mlflow.end_run()
    except Exception as e:
        print(f"Warning: Could not log to MLFlow: {e}")


def _log_train_validation_to_mlflow(experiment_name, all_results):
    """Log train-validation summary to MLFlow."""
    try:
        mlflow.set_experiment(f"{experiment_name}_summary")
        with mlflow.start_run(run_name=f"train_validation_summary"):
            mlflow.set_tag("type", "train_validation_summary")

            # Log aggregate metrics for each model
            for model_name, model_data in all_results.get("models", {}).items():
                agg = model_data.get("aggregate_metrics", {})
                if agg:
                    # Log validation accuracy
                    val_acc = agg.get("validation_accuracy", {})
                    if val_acc:
                        mlflow.log_metric(
                            f"{model_name}_val_acc_mean", float(val_acc.get("mean", 0))
                        )
                        mlflow.log_metric(
                            f"{model_name}_val_acc_ci_low",
                            float(val_acc.get("ci_low", 0)),
                        )
                        mlflow.log_metric(
                            f"{model_name}_val_acc_ci_high",
                            float(val_acc.get("ci_high", 0)),
                        )

                    # Log validation loss
                    val_loss = agg.get("validation_loss", {})
                    if val_loss:
                        mlflow.log_metric(
                            f"{model_name}_val_loss_mean",
                            float(val_loss.get("mean", 0)),
                        )

                    # Log macro metrics
                    macro_avg = agg.get("macro_avg", {})
                    if macro_avg:
                        for metric_name, metric_vals in macro_avg.items():
                            if isinstance(metric_vals, dict):
                                mlflow.log_metric(
                                    f"{model_name}_macro_{metric_name}_mean",
                                    float(metric_vals.get("mean", 0)),
                                )

            # Log complete results as JSON artifact
            mlflow.log_dict(all_results, "train_validation_results.json")

            mlflow.end_run()
    except Exception as e:
        print(f"Warning: Could not log to MLFlow: {e}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
