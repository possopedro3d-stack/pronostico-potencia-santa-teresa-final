# -*- coding: utf-8 -*-
"""
Experimento corregido de pronóstico para la central Santa Teresa.

Objetivo:
- Recalcular la partición fija para las ventanas de 4, 12 y 24 registros.
- Mantener una matriz común de 5007 observaciones útiles.
- Aplicar límites temporales comunes para G1 y G2.
- Corregir la persistencia: último valor disponible -> siguiente intervalo.
- Entrenar Random Forest y XGBoost como modelos globales.
- Entrenar LSTM de forma independiente para G1 y G2.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

DATA_FILE = Path("santa_teresa_serie_larga_5057.csv")
OUTPUT_DIR = Path("outputs_experimento_corregido")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "fecha_hora"
UNIT_COL = "unidad"
TARGET_COL = "potencia_efectiva"
WINDOWS = [4, 12, 24]

TRAIN_END = pd.Timestamp("2025-02-23 21:30:00")
VALIDATION_END = pd.Timestamp("2025-03-13 20:45:00")


def read_csv_robust(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def calculate_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mape(y_true, y_pred),
    }


def prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    required = {DATE_COL, UNIT_COL, TARGET_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out[UNIT_COL] = out[UNIT_COL].astype(str).str.strip().str.upper()

    out = out.dropna(subset=[DATE_COL, UNIT_COL, TARGET_COL])
    out = out[out[UNIT_COL].isin(["G1", "G2"])]
    out = out.drop_duplicates(subset=[DATE_COL, UNIT_COL], keep="first")
    out = out.sort_values([UNIT_COL, DATE_COL]).reset_index(drop=True)

    out["unidad_id"] = out[UNIT_COL].map({"G1": 1, "G2": 2}).astype(int)
    out["hora"] = out[DATE_COL].dt.hour
    out["minuto"] = out[DATE_COL].dt.minute
    out["dia_semana"] = out[DATE_COL].dt.dayofweek
    out["mes"] = out[DATE_COL].dt.month
    return out


def build_common_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una matriz común de 5007 observaciones.

    La matriz exige 24 registros históricos disponibles y un valor futuro.
    De este modo, las ventanas de 4, 12 y 24 se comparan sobre las mismas
    observaciones y los mismos límites temporales.
    """
    frames = []

    for unit, group in df.groupby(UNIT_COL, sort=False):
        data = group.sort_values(DATE_COL).copy()

        data["potencia_actual"] = data[TARGET_COL]

        for lag in range(1, 25):
            data[f"lag_{lag}"] = data[TARGET_COL].shift(lag)

        # Medias móviles que terminan en el instante actual.
        data["media_movil_4"] = data[TARGET_COL].rolling(4).mean()
        data["media_movil_12"] = data[TARGET_COL].rolling(12).mean()
        data["media_movil_24"] = data[TARGET_COL].rolling(24).mean()

        data["y"] = data[TARGET_COL].shift(-1)
        data["fecha_objetivo"] = data[DATE_COL].shift(-1)
        frames.append(data)

    panel = pd.concat(frames, ignore_index=True)

    # lag_24 fija el calentamiento común de 24 observaciones;
    # y elimina el último registro de cada unidad.
    panel = (
        panel.dropna(subset=["lag_24", "y", "fecha_objetivo"])
        .sort_values([DATE_COL, UNIT_COL])
        .reset_index(drop=True)
    )
    return panel


def feature_columns(window: int) -> list[str]:
    """
    Cada ventana usa exactamente 'window' valores de potencia:
    potencia actual + window-1 retardos.
    """
    cols = ["potencia_actual"]
    cols += [f"lag_{i}" for i in range(1, window)]

    if window >= 4:
        cols.append("media_movil_4")
    if window >= 12:
        cols.append("media_movil_12")
    if window >= 24:
        cols.append("media_movil_24")

    cols += ["unidad_id", "hora", "minuto", "dia_semana", "mes"]
    return cols


def build_lstm_sequences(
    base_df: pd.DataFrame,
    window: int,
    eligible_pairs: set[tuple[pd.Timestamp, str]],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Genera secuencias independientes de potencia efectiva para G1 y G2.

    La secuencia contiene exactamente 'window' observaciones y termina en
    la fecha de origen. El objetivo corresponde al intervalo siguiente.
    """
    result = {}

    for unit, group in base_df.groupby(UNIT_COL, sort=False):
        g = group.sort_values(DATE_COL).reset_index(drop=True)
        values = g[TARGET_COL].to_numpy(dtype=float)

        X, y, origin_dates = [], [], []

        for target_idx in range(window, len(g)):
            origin_idx = target_idx - 1
            origin_date = pd.Timestamp(g.loc[origin_idx, DATE_COL])

            if (origin_date, unit) not in eligible_pairs:
                continue

            sequence = values[target_idx - window:target_idx]
            X.append(sequence.reshape(window, 1))
            y.append(values[target_idx])
            origin_dates.append(origin_date)

        result[unit] = (
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(origin_dates, dtype="datetime64[ns]"),
        )

    return result


def make_lstm(window: int) -> Sequential:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = Sequential(
        [
            LSTM(64, input_shape=(window, 1)),
            Dropout(0.15),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def add_metric_rows(
    rows: list[dict],
    predictions: pd.DataFrame,
    model_name: str,
    window: int,
    elapsed: float,
) -> None:
    for unit in ["G1", "G2"]:
        sub = predictions[predictions[UNIT_COL] == unit]
        metrics = calculate_metrics(sub["real"], sub["prediccion"])
        rows.append(
            {
                "modelo": model_name,
                "ventana": window,
                "unidad": unit,
                **metrics,
                "tiempo_segundos": elapsed,
                "inicio_prueba": sub[DATE_COL].min(),
                "fin_prueba": sub[DATE_COL].max(),
                "n_observaciones": len(sub),
            }
        )

    metrics = calculate_metrics(predictions["real"], predictions["prediccion"])
    rows.append(
        {
            "modelo": model_name,
            "ventana": window,
            "unidad": "Agregado",
            **metrics,
            "tiempo_segundos": elapsed,
            "inicio_prueba": predictions[DATE_COL].min(),
            "fin_prueba": predictions[DATE_COL].max(),
            "n_observaciones": len(predictions),
        }
    )


def run_persistence(
    panel: pd.DataFrame,
    test_mask: pd.Series,
) -> tuple[pd.DataFrame, float]:
    test = panel.loc[test_mask].copy()

    start = time.perf_counter()
    y_pred = test["potencia_actual"].to_numpy()
    elapsed = time.perf_counter() - start

    pred = test[[DATE_COL, "fecha_objetivo", UNIT_COL]].copy()
    pred["real"] = test["y"].to_numpy()
    pred["prediccion"] = y_pred
    pred["modelo"] = "Línea base de persistencia"
    return pred, elapsed


def run_tabular_model(
    model,
    model_name: str,
    panel: pd.DataFrame,
    features: list[str],
    train_mask: pd.Series,
    test_mask: pd.Series,
) -> tuple[pd.DataFrame, float]:
    train = panel.loc[train_mask].copy()
    test = panel.loc[test_mask].copy()

    start = time.perf_counter()
    model.fit(train[features], train["y"])
    y_pred = model.predict(test[features])
    elapsed = time.perf_counter() - start

    pred = test[[DATE_COL, "fecha_objetivo", UNIT_COL]].copy()
    pred["real"] = test["y"].to_numpy()
    pred["prediccion"] = y_pred
    pred["modelo"] = model_name
    return pred, elapsed


def run_lstm_independent(
    sequences: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    window: int,
) -> tuple[pd.DataFrame, float]:
    predictions = []
    total_elapsed = 0.0

    for unit in ["G1", "G2"]:
        X, y, dates = sequences[unit]
        dates_pd = pd.to_datetime(dates)

        train_idx = np.where(dates_pd <= TRAIN_END)[0]
        val_idx = np.where(
            (dates_pd > TRAIN_END) & (dates_pd <= VALIDATION_END)
        )[0]
        test_idx = np.where(dates_pd > VALIDATION_END)[0]

        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                f"Partición vacía para LSTM {unit}, ventana {window}. "
                f"Train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
            )

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        sx = MinMaxScaler()
        sy = MinMaxScaler()

        sx.fit(X_train.reshape(-1, 1))
        sy.fit(y_train.reshape(-1, 1))

        X_train_s = sx.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
        X_val_s = sx.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
        X_test_s = sx.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

        y_train_s = sy.transform(y_train.reshape(-1, 1)).ravel()
        y_val_s = sy.transform(y_val.reshape(-1, 1)).ravel()

        model = make_lstm(window)
        early = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        start = time.perf_counter()
        model.fit(
            X_train_s,
            y_train_s,
            validation_data=(X_val_s, y_val_s),
            epochs=80,
            batch_size=32,
            callbacks=[early],
            verbose=0,
            shuffle=False,
        )
        y_pred_scaled = model.predict(X_test_s, verbose=0).ravel()
        elapsed = time.perf_counter() - start
        total_elapsed += elapsed

        y_pred = sy.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).ravel()

        pred = pd.DataFrame(
            {
                DATE_COL: dates_pd[test_idx],
                "fecha_objetivo": dates_pd[test_idx] + pd.Timedelta(minutes=15),
                UNIT_COL: unit,
                "real": y_test,
                "prediccion": y_pred,
                "modelo": "LSTM",
            }
        )
        predictions.append(pred)

    final = (
        pd.concat(predictions, ignore_index=True)
        .sort_values([DATE_COL, UNIT_COL])
        .reset_index(drop=True)
    )
    return final, total_elapsed


def make_models():
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=SEED,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    xgb = XGBRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )
    return rf, xgb


def save_summary_plot(aggregate: pd.DataFrame, metric: str) -> None:
    pivot = aggregate.pivot(
        index="ventana",
        columns="modelo",
        values=metric,
    ).sort_index()

    ax = pivot.plot(marker="o", figsize=(10, 6))
    ax.set_xlabel("Ventana temporal (registros)")
    ax.set_ylabel(f"{metric} ({'%' if metric == 'MAPE' else 'MW'})")
    ax.set_title(f"{metric} agregado por modelo y ventana temporal")
    plt.xticks(WINDOWS)
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{metric.lower()}_agregado_por_ventana.png",
        dpi=180,
    )
    plt.close()


def main():
    print("=" * 78)
    print("EXPERIMENTO CORREGIDO - VENTANAS 4, 12 Y 24")
    print("=" * 78)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"No se encuentra {DATA_FILE}. Coloque el script junto al CSV."
        )

    base = prepare_base(read_csv_robust(DATA_FILE))
    panel = build_common_panel(base)

    print(f"Base original: {len(base)} registros")
    print(f"Matriz supervisada común: {len(panel)} observaciones útiles")
    print(f"G1 útiles: {(panel[UNIT_COL] == 'G1').sum()}")
    print(f"G2 útiles: {(panel[UNIT_COL] == 'G2').sum()}")
    print(f"Período de origen: {panel[DATE_COL].min()} a {panel[DATE_COL].max()}")

    if len(base) != 5057:
        print(f"ADVERTENCIA: se esperaban 5057 registros y se encontraron {len(base)}.")
    if len(panel) != 5007:
        raise ValueError(
            f"La matriz común debe contener 5007 observaciones, "
            f"pero contiene {len(panel)}."
        )

    train_mask = panel[DATE_COL] <= TRAIN_END
    val_mask = (
        (panel[DATE_COL] > TRAIN_END)
        & (panel[DATE_COL] <= VALIDATION_END)
    )
    test_mask = panel[DATE_COL] > VALIDATION_END

    partition = pd.DataFrame(
        {
            "conjunto": ["Entrenamiento", "Validación", "Prueba"],
            "inicio": [
                panel.loc[train_mask, DATE_COL].min(),
                panel.loc[val_mask, DATE_COL].min(),
                panel.loc[test_mask, DATE_COL].min(),
            ],
            "fin": [
                panel.loc[train_mask, DATE_COL].max(),
                panel.loc[val_mask, DATE_COL].max(),
                panel.loc[test_mask, DATE_COL].max(),
            ],
            "n_observaciones": [
                int(train_mask.sum()),
                int(val_mask.sum()),
                int(test_mask.sum()),
            ],
        }
    )
    partition["porcentaje"] = (
        partition["n_observaciones"] / len(panel) * 100
    )
    partition.to_csv(
        OUTPUT_DIR / "particion_temporal_comun.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nPartición temporal común:")
    print(partition.to_string(index=False))

    eligible_pairs = set(
        zip(pd.to_datetime(panel[DATE_COL]), panel[UNIT_COL].astype(str))
    )

    all_metrics = []
    all_predictions = []

    for window in WINDOWS:
        print("\n" + "-" * 78)
        print(f"PROCESANDO VENTANA DE {window} REGISTROS")
        print("-" * 78)

        features = feature_columns(window)
        print(f"Variables tabulares: {features}")

        # Línea base
        pred, elapsed = run_persistence(panel, test_mask)
        pred["ventana"] = window
        all_predictions.append(pred)
        add_metric_rows(
            all_metrics,
            pred,
            "Línea base de persistencia",
            window,
            elapsed,
        )

        # Random Forest global
        rf, xgb = make_models()

        pred, elapsed = run_tabular_model(
            rf,
            "Random Forest",
            panel,
            features,
            train_mask,
            test_mask,
        )
        pred["ventana"] = window
        all_predictions.append(pred)
        add_metric_rows(
            all_metrics,
            pred,
            "Random Forest",
            window,
            elapsed,
        )

        # XGBoost global
        pred, elapsed = run_tabular_model(
            xgb,
            "XGBoost",
            panel,
            features,
            train_mask,
            test_mask,
        )
        pred["ventana"] = window
        all_predictions.append(pred)
        add_metric_rows(
            all_metrics,
            pred,
            "XGBoost",
            window,
            elapsed,
        )

        # LSTM independiente para G1 y G2
        sequences = build_lstm_sequences(base, window, eligible_pairs)
        pred, elapsed = run_lstm_independent(sequences, window)
        pred["ventana"] = window
        all_predictions.append(pred)
        add_metric_rows(
            all_metrics,
            pred,
            "LSTM",
            window,
            elapsed,
        )

        print(f"Ventana {window} finalizada.")

    metrics = pd.DataFrame(all_metrics)
    predictions = pd.concat(all_predictions, ignore_index=True)

    metrics.to_csv(
        OUTPUT_DIR / "metricas_corregidas_por_unidad_y_ventana.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        OUTPUT_DIR / "predicciones_corregidas_por_ventana.csv",
        index=False,
        encoding="utf-8-sig",
    )

    aggregate = metrics[metrics["unidad"] == "Agregado"].copy()
    aggregate = aggregate.sort_values(["ventana", "MAE", "RMSE"])
    aggregate.to_csv(
        OUTPUT_DIR / "metricas_agregadas_corregidas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Mejor configuración de cada modelo.
    best_by_model_rows = []
    for model, group in aggregate.groupby("modelo", sort=False):
        ordered = group.sort_values(["MAE", "RMSE", "MAPE"])
        best_by_model_rows.append(ordered.iloc[0])

    best_by_model = pd.DataFrame(best_by_model_rows).reset_index(drop=True)
    best_by_model.to_csv(
        OUTPUT_DIR / "mejor_configuracion_por_modelo.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Mejor resultado dentro de cada ventana.
    best_by_window_rows = []
    for window, group in aggregate.groupby("ventana", sort=True):
        ordered = group.sort_values(["MAE", "RMSE", "MAPE"])
        best_by_window_rows.append(ordered.iloc[0])

    best_by_window = pd.DataFrame(best_by_window_rows).reset_index(drop=True)
    best_by_window.to_csv(
        OUTPUT_DIR / "mejor_resultado_por_ventana.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for metric in ["MAE", "RMSE", "MAPE"]:
        save_summary_plot(aggregate, metric)

    print("\n" + "=" * 78)
    print("MÉTRICAS AGREGADAS CORREGIDAS")
    print("=" * 78)
    print(
        aggregate[
            ["modelo", "ventana", "MAE", "RMSE", "MAPE", "tiempo_segundos"]
        ].to_string(index=False)
    )

    print("\nArchivos generados en:")
    print(OUTPUT_DIR.resolve())
    print("\nEjecución finalizada correctamente.")


if __name__ == "__main__":
    main()
