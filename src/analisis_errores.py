# -*- coding: utf-8 -*-
"""
Genera los insumos corregidos del apartado 5.12:
- Tabla 25: principales errores absolutos.
- Resumen de concentración de errores por unidad.
- Resumen del tramo crítico de G2.
- Figura 10: comparación de modelos en el tramo crítico de G2.

Entrada:
predicciones_corregidas_por_ventana.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

INPUT_FILE = Path("predicciones_corregidas_por_ventana.csv")
OUTPUT_DIR = Path("analisis_errores_corregido")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "fecha_objetivo"
MAX_GAP = pd.Timedelta(minutes=15)

CONFIGURACIONES = {
    "Línea base de persistencia": 4,
    "Random Forest": 24,
    "XGBoost": 4,
    "LSTM": 4,
}


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
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")

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
            "unidad",
            "real",
            "prediccion",
            "modelo",
            "ventana",
        ]
    )
    out = out[out["unidad"].isin(["G1", "G2"])]

    bloques = []
    for modelo, ventana in CONFIGURACIONES.items():
        sub = out[
            (out["modelo"] == modelo)
            & (out["ventana"] == ventana)
        ].copy()
        if sub.empty:
            raise ValueError(
                f"No se encontraron predicciones para {modelo}, "
                f"ventana {ventana}."
            )
        bloques.append(sub)

    selected = pd.concat(bloques, ignore_index=True)
    selected["error_absoluto"] = (
        selected["real"] - selected["prediccion"]
    ).abs()
    selected["error_porcentual"] = np.where(
        selected["real"].abs() > 1e-12,
        selected["error_absoluto"] / selected["real"].abs() * 100,
        np.nan,
    )
    return selected.sort_values(
        [DATE_COL, "unidad", "modelo"]
    ).reset_index(drop=True)


def guardar_tabla_25(df: pd.DataFrame) -> pd.DataFrame:
    top = (
        df.sort_values("error_absoluto", ascending=False)
        .head(10)
        .copy()
    )
    top.insert(0, "posición", range(1, len(top) + 1))
    top["ventana"] = top.apply(
        lambda r: "4, 12 y 24"
        if r["modelo"] == "Línea base de persistencia"
        else int(r["ventana"]),
        axis=1,
    )

    columnas = [
        "posición",
        DATE_COL,
        "unidad",
        "modelo",
        "ventana",
        "real",
        "prediccion",
        "error_absoluto",
        "error_porcentual",
    ]
    top[columnas].to_csv(
        OUTPUT_DIR / "tabla_25_principales_errores_absolutos.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return top[columnas]


def guardar_resumen_unidad(df: pd.DataFrame) -> pd.DataFrame:
    resumen = (
        df.groupby("unidad", as_index=False)
        .agg(
            número_predicciones=("error_absoluto", "size"),
            error_absoluto_medio_MW=("error_absoluto", "mean"),
            error_absoluto_máximo_MW=("error_absoluto", "max"),
            percentil_95_error_MW=("error_absoluto", lambda x: x.quantile(0.95)),
            error_porcentual_medio=("error_porcentual", "mean"),
        )
    )

    top20 = (
        df.nlargest(20, "error_absoluto")
        .groupby("unidad")
        .size()
        .rename("presencia_en_top_20")
        .reset_index()
    )

    resumen = resumen.merge(top20, on="unidad", how="left")
    resumen["presencia_en_top_20"] = (
        resumen["presencia_en_top_20"].fillna(0).astype(int)
    )
    resumen.to_csv(
        OUTPUT_DIR / "resumen_concentracion_errores_por_unidad.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return resumen


def insertar_cortes(sub: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    sub = sub.sort_values(DATE_COL).reset_index(drop=True)
    filas = []

    for i, row in sub.iterrows():
        if i > 0:
            anterior = sub.loc[i - 1, DATE_COL]
            actual = row[DATE_COL]
            if actual - anterior > MAX_GAP:
                corte = {DATE_COL: anterior + (actual - anterior) / 2}
                for col in value_cols:
                    corte[col] = np.nan
                filas.append(corte)

        nueva = {DATE_COL: row[DATE_COL]}
        for col in value_cols:
            nueva[col] = row[col]
        filas.append(nueva)

    return pd.DataFrame(filas)


def generar_figura_10(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame]:
    g2 = df[df["unidad"] == "G2"].copy()

    fila_max = g2.loc[g2["error_absoluto"].idxmax()]
    fecha_critica = pd.Timestamp(fila_max[DATE_COL])

    fechas = np.array(sorted(g2[DATE_COL].unique()))
    idx = int(np.where(fechas == np.datetime64(fecha_critica))[0][0])

    inicio_idx = max(0, idx - 30)
    fin_idx = min(len(fechas), idx + 31)
    fechas_tramo = fechas[inicio_idx:fin_idx]

    tramo = g2[g2[DATE_COL].isin(fechas_tramo)].copy()

    real = (
        tramo[[DATE_COL, "real"]]
        .drop_duplicates(subset=[DATE_COL])
        .sort_values(DATE_COL)
    )

    pivot = (
        tramo.pivot_table(
            index=DATE_COL,
            columns="modelo",
            values="prediccion",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(DATE_COL)
    )

    grafico = real.merge(pivot, on=DATE_COL, how="left")
    columnas_valor = ["real"] + list(CONFIGURACIONES.keys())
    grafico_cortado = insertar_cortes(grafico, columnas_valor)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        grafico_cortado[DATE_COL],
        grafico_cortado["real"],
        linewidth=1.8,
        label="Valor real G2",
    )

    for modelo in CONFIGURACIONES:
        ax.plot(
            grafico_cortado[DATE_COL],
            grafico_cortado[modelo],
            linestyle="--",
            linewidth=1.3,
            label=modelo,
        )

    ax.axvline(
        fecha_critica,
        linestyle=":",
        linewidth=1.4,
        label="Instante de mayor error",
    )
    ax.set_title("Tramo crítico de G2: valores reales y predicciones")
    ax.set_xlabel("Fecha y hora")
    ax.set_ylabel("Potencia efectiva (MW)")
    ax.set_ylim(0, 15)
    ax.grid(True, alpha=0.25)
    ax.legend()

    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.DateFormatter("%d-%m-%Y\n%H:%M")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "figura_10_tramo_critico_g2.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    resumen_critico = tramo[
        tramo[DATE_COL] == fecha_critica
    ][
        [
            DATE_COL,
            "unidad",
            "modelo",
            "ventana",
            "real",
            "prediccion",
            "error_absoluto",
            "error_porcentual",
        ]
    ].sort_values("error_absoluto", ascending=False)

    resumen_critico.to_csv(
        OUTPUT_DIR / "resumen_instante_critico_g2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    grafico.to_csv(
        OUTPUT_DIR / "datos_figura_10_tramo_critico_g2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return fecha_critica, resumen_critico


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No se encuentra {INPUT_FILE}. Coloque este script en la "
            "misma carpeta que el CSV."
        )

    df = preparar_datos(leer_csv(INPUT_FILE))

    tabla_25 = guardar_tabla_25(df)
    resumen_unidad = guardar_resumen_unidad(df)
    fecha_critica, resumen_critico = generar_figura_10(df)

    print("=" * 78)
    print("TABLA 25. PRINCIPALES ERRORES ABSOLUTOS")
    print("=" * 78)
    print(tabla_25.to_string(index=False))

    print("\n" + "=" * 78)
    print("CONCENTRACIÓN DE ERRORES POR UNIDAD")
    print("=" * 78)
    print(resumen_unidad.to_string(index=False))

    print("\n" + "=" * 78)
    print(f"INSTANTE CRÍTICO DE G2: {fecha_critica}")
    print("=" * 78)
    print(resumen_critico.to_string(index=False))

    print("\nArchivos generados en:")
    print(OUTPUT_DIR.resolve())
    print("\nProceso finalizado correctamente.")


if __name__ == "__main__":
    main()
