# -*- coding: utf-8 -*-
"""
Genera una versión corregida de las Figuras 2 a 9.

Correcciones:
- Interrumpe las líneas cuando existe un salto temporal superior a 15 minutos.
- Evita mostrar tendencias diagonales artificiales entre bloques separados.
- Elimina la numeración "Figura X" del título interno de la imagen.
- Mantiene escalas comunes:
    * Gráficas generales: 0 a 15 MW.
    * Gráficas de detalle: 8 a 14,5 MW.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_FILE = Path("predicciones_corregidas_por_ventana.csv")
OUTPUT_DIR = Path("figuras_predicciones_corregidas_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "fecha_objetivo"
MAX_GAP = pd.Timedelta(minutes=15)

CONFIGURACIONES = [
    {
        "modelo": "Línea base de persistencia",
        "ventana": 4,
        "fig_general": 2,
        "fig_detalle": 3,
        "nombre": "linea_base_persistencia",
        "titulo": "Línea base de persistencia",
    },
    {
        "modelo": "Random Forest",
        "ventana": 24,
        "fig_general": 4,
        "fig_detalle": 5,
        "nombre": "random_forest_w24",
        "titulo": "Random Forest, ventana de 24 registros",
    },
    {
        "modelo": "XGBoost",
        "ventana": 4,
        "fig_general": 6,
        "fig_detalle": 7,
        "nombre": "xgboost_w4",
        "titulo": "XGBoost, ventana de 4 registros",
    },
    {
        "modelo": "LSTM",
        "ventana": 4,
        "fig_general": 8,
        "fig_detalle": 9,
        "nombre": "lstm_w4",
        "titulo": "LSTM, ventana de 4 registros",
    },
]


def leer_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")

    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    requeridas = {
        DATE_COL,
        "unidad",
        "real",
        "prediccion",
        "modelo",
        "ventana",
    }
    faltantes = requeridas.difference(df.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas obligatorias: {sorted(faltantes)}"
        )

    out = df.copy()
    out["modelo"] = out["modelo"].astype(str).str.strip().replace(
        {
            "Baseline persistencia": "Línea base de persistencia",
            "Baseline de persistencia": "Línea base de persistencia",
            "Linea base de persistencia": "Línea base de persistencia",
        }
    )
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out["ventana"] = pd.to_numeric(out["ventana"], errors="coerce")
    out["real"] = pd.to_numeric(out["real"], errors="coerce")
    out["prediccion"] = pd.to_numeric(out["prediccion"], errors="coerce")
    out["unidad"] = out["unidad"].astype(str).str.strip().str.upper()

    out = out.dropna(
        subset=[
            DATE_COL,
            "ventana",
            "real",
            "prediccion",
            "modelo",
            "unidad",
        ]
    )
    out = out[out["unidad"].isin(["G1", "G2"])]
    return out.sort_values([DATE_COL, "unidad"]).reset_index(drop=True)


def insertar_cortes_temporales(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Inserta filas con NaN cuando el salto entre observaciones supera 15 minutos.
    Matplotlib interrumpe la línea en esas filas y evita unir bloques separados.
    """
    sub = sub.sort_values(DATE_COL).reset_index(drop=True)
    filas = []

    for i, row in sub.iterrows():
        if i > 0:
            fecha_anterior = sub.loc[i - 1, DATE_COL]
            fecha_actual = row[DATE_COL]

            if fecha_actual - fecha_anterior > MAX_GAP:
                fecha_intermedia = fecha_anterior + (
                    fecha_actual - fecha_anterior
                ) / 2
                filas.append(
                    {
                        DATE_COL: fecha_intermedia,
                        "real": np.nan,
                        "prediccion": np.nan,
                    }
                )

        filas.append(
            {
                DATE_COL: row[DATE_COL],
                "real": row["real"],
                "prediccion": row["prediccion"],
            }
        )

    return pd.DataFrame(filas)


def graficar(
    datos: pd.DataFrame,
    titulo_modelo: str,
    archivo: Path,
    escala_detalle: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    for unidad in ["G1", "G2"]:
        sub = datos[datos["unidad"] == unidad]
        if sub.empty:
            continue

        trazado = insertar_cortes_temporales(sub)

        ax.plot(
            trazado[DATE_COL],
            trazado["real"],
            linewidth=1.4,
            label=f"Real {unidad}",
        )
        ax.plot(
            trazado[DATE_COL],
            trazado["prediccion"],
            linewidth=1.2,
            linestyle="--",
            label=f"Predicción {unidad}",
        )

    if escala_detalle:
        ax.set_ylim(8, 14.5)
        ax.set_title(
            f"{titulo_modelo}: detalle de la escala operativa habitual"
        )
    else:
        ax.set_ylim(0, 15)
        ax.set_title(f"{titulo_modelo}: valores reales y predichos")

    ax.set_xlabel("Fecha y hora")
    ax.set_ylabel("Potencia efectiva (MW)")
    ax.legend()
    ax.grid(True, alpha=0.25)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.DateFormatter("%d-%m-%Y")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(archivo, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No se encuentra {INPUT_FILE}. El script debe estar en la "
            "misma carpeta que el CSV."
        )

    df = preparar_datos(leer_csv(INPUT_FILE))

    for cfg in CONFIGURACIONES:
        datos = df[
            (df["modelo"] == cfg["modelo"])
            & (df["ventana"] == cfg["ventana"])
        ].copy()

        if datos.empty:
            disponibles = (
                df[["modelo", "ventana"]]
                .drop_duplicates()
                .sort_values(["modelo", "ventana"])
            )
            raise ValueError(
                f"No se encontraron datos para {cfg['modelo']} "
                f"con ventana {cfg['ventana']}.\n"
                f"Configuraciones disponibles:\n"
                f"{disponibles.to_string(index=False)}"
            )

        general = OUTPUT_DIR / (
            f"figura_{cfg['fig_general']}_{cfg['nombre']}_general_v2.png"
        )
        detalle = OUTPUT_DIR / (
            f"figura_{cfg['fig_detalle']}_{cfg['nombre']}_detalle_v2.png"
        )

        graficar(
            datos,
            cfg["titulo"],
            general,
            escala_detalle=False,
        )
        graficar(
            datos,
            cfg["titulo"],
            detalle,
            escala_detalle=True,
        )

        print(f"Generadas: {general.name} y {detalle.name}")

    print("\nFiguras guardadas en:")
    print(OUTPUT_DIR.resolve())
    print("\nProceso finalizado correctamente.")


if __name__ == "__main__":
    main()
