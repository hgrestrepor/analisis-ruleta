"""Orquestación del análisis por lotes.

El dashboard consulta el estado mientras un hilo de fondo recorre las capturas.
El progreso se expone como un snapshot serializable para que la API lo devuelva
tal cual.
"""

from __future__ import annotations

import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from core import basedatos as BD
from core import estadisticas as E
from core import vision

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_CAPTURAS = RAIZ / "imagenes" / "capturas"
EXTENSIONES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Analizador:
    """Mantiene un único análisis en curso y su progreso."""

    def __init__(self) -> None:
        self._candado = threading.Lock()
        self._hilo: threading.Thread | None = None
        self.estado: dict = {
            "activo": False,
            "total": 0,
            "procesadas": 0,
            "actual": None,
            "log": [],
            "iniciado_en": None,
            "terminado_en": None,
            "errores": 0,
            "tiradas_nuevas": 0,
        }

    # ---------------------------------------------------------------- estado
    def snapshot(self) -> dict:
        with self._candado:
            datos = dict(self.estado)
            datos["log"] = list(self.estado["log"])
            datos["pct"] = (
                round(100.0 * datos["procesadas"] / datos["total"], 1)
                if datos["total"]
                else 0.0
            )
            return datos

    def _log(self, linea: str) -> None:
        with self._candado:
            self.estado["log"].append(linea)
            del self.estado["log"][:-200]

    # ---------------------------------------------------------------- inicio
    def hay_capturas(self) -> bool:
        return any(
            p.suffix.lower() in EXTENSIONES for p in CARPETA_CAPTURAS.glob("*")
        )

    def contar_capturas(self) -> int:
        return sum(
            1 for p in CARPETA_CAPTURAS.glob("*") if p.suffix.lower() in EXTENSIONES
        )

    def iniciar(self, reiniciar_historial: bool = False) -> bool:
        with self._candado:
            if self._hilo and self._hilo.is_alive():
                return False
            # lo que ya no está en disco no debe seguir contando como captura
            podadas = BD.podar_capturas_inexistentes(CARPETA_CAPTURAS)
            if podadas:
                self._log(f"Limpieza: {podadas} captura(s) borrada(s) del disco, "
                          "se quitaron del contador")
            self.estado = {
                "activo": True,
                "total": self.contar_capturas(),
                "procesadas": 0,
                "actual": None,
                "log": [],
                "iniciado_en": _ahora(),
                "terminado_en": None,
                "errores": 0,
                "tiradas_nuevas": 0,
            }
            hilo = threading.Thread(
                target=self._trabajar,
                args=(reiniciar_historial,),
                daemon=True,
            )
            self._hilo = hilo
        hilo.start()
        return True

    # ------------------------------------------------------------- worker
    def _trabajar(self, reiniciar: bool) -> None:
        try:
            if reiniciar:
                BD.limpiar()
                self._log("Historial anterior borrado.")

            cfg = vision.cargar_config()
            rutas = sorted(
                p for p in CARPETA_CAPTURAS.glob("*") if p.suffix.lower() in EXTENSIONES
            )
            inicio = time.time()
            nuevas = 0

            for ruta in rutas:
                with self._candado:
                    self.estado["actual"] = ruta.name
                self._analizar_una(ruta, cfg)
                with self._candado:
                    self.estado["procesadas"] += 1
                time.sleep(0.0)  # cede el hilo para que la UI refresque

            nuevas = BD.resumen_capturas()["tiradas"]
            with self._candado:
                self.estado["tiradas_nuevas"] = nuevas
                self.estado["actual"] = None
                self.estado["terminado_en"] = _ahora()
                self.estado["activo"] = False
            elapsed = time.time() - inicio
            self._log(
                f"Listo: {len(rutas)} capturas en {elapsed:.1f}s "
                f"({nuevas} tiradas en el historial)."
            )
        except Exception:
            self._log("Fallo grave:\n" + traceback.format_exc())
            with self._candado:
                self.estado["activo"] = False
                self.estado["terminado_en"] = _ahora()

    def _analizar_una(self, ruta: Path, cfg: dict) -> None:
        # reprocesar una captura reemplaza sus filas anteriores en vez de duplicarlas
        BD.borrar_por_archivo(ruta.name)
        try:
            res = vision.analizar_imagen(ruta, cfg)
        except Exception as exc:
            with self._candado:
                self.estado["errores"] += 1
            self._log(f"{ruta.name}: error al analizar ({exc}).")
            BD.registrar_captura(
                archivo=ruta.name,
                ruta=str(ruta),
                analizada_en=_ahora(),
                modo="-",
                n_detectadas=0,
                n_reconocidas=0,
                n_dudas=0,
                n_conflicto=0,
                confianza=0.0,
                rejilla=None,
                estado="error",
                error=str(exc),
            )
            return

        filas = []
        for celda in res.celdas:
            if celda.get("numero") is None:
                continue
            filas.append(
                (
                    int(celda["numero"]),
                    float(celda.get("confianza") or 0.0),
                    celda.get("color_esperado") or "",
                    "duda" if celda.get("confianza", 0) < cfg.get("umbral_confianza", 0.62) else "ok",
                )
            )

        BD.registrar_captura(
            archivo=ruta.name,
            ruta=str(ruta),
            analizada_en=_ahora(),
            modo=res.modo,
            n_detectadas=res.n_detectadas,
            n_reconocidas=res.n_reconocidas,
            n_dudas=res.n_dudas,
            n_conflicto=res.n_conflicto_color,
            confianza=res.confianza_promedio,
            rejilla=res.rejilla,
            estado="ok" if res.n_reconocidas else "vacia",
            error="; ".join(res.errores) if res.errores else None,
            tiradas=filas,
        )

        # Si se leyó menos de tres cuartas partes de lo detectado, la captura no
        # encaja con lo que el detector esperaba: hay que decirlo, porque el conteo
        # de dígitos parece correcto mientras el resultado no lo es.
        leido_bien = res.n_detectadas == 0 or res.n_reconocidas / res.n_detectadas >= 0.75
        marca = "!" if (res.necesita_calibracion(cfg) or not leido_bien) else " "
        if res.modo == "cuadros":
            detalle = ""
        elif not leido_bien:
            detalle = " · se leyó menos de 3/4 de lo detectado: revisa el recorte"
        else:
            detalle = (
                f" · revisar rejilla (residuo {res.rejilla['residuo_rel']:.2f})"
                if res.necesita_calibracion(cfg) and res.rejilla
                else ""
            )
        self._log(
            f"{marca} {ruta.name}: {res.n_reconocidas}/{res.n_detectadas} "
            f"dígitos, modo {res.modo}, confianza {res.confianza_promedio:.2f}"
            + detalle
        )


ANALIZADOR = Analizador()


# ------------------------------------------------------------------ métricas
def metricas_completas(limite_capturas: int = 50) -> dict:
    """Todo lo que el dashboard necesita, en una sola respuesta."""
    tiradas = BD.todas_las_tiradas()
    historial = BD.resumen_capturas()
    resumen = E.resumen_desde_tiradas(tiradas)
    resumen["capturas"] = historial["capturas_ok"]
    esquema = E.esquema_tablas("docenas")
    return {
        "esquema": esquema,
        "resumen": resumen,
        "capturas": historial,
        "capturas_disponibles": ANALIZADOR.contar_capturas(),
        "ultimas_capturas": BD.ultimas_capturas(limite_capturas),
        "ajustes_rejilla": BD.listar_ajustes(),
        "alternancia": resumen["alternancia_docenas"],
        "progreso": ANALIZADOR.snapshot(),
        "sin_datos": not tiradas,
    }


def rellenar_tabla(esquema: dict, resumen: dict) -> dict:
    """Rellena los `None` de las tablas del esquema con los valores reales."""
    n = resumen["total"]
    esp = resumen["esperado_numero"]

    globals_ = esquema["resumen_global"]
    globals_["nota"] = (
        f"Datos acumulados en SQLite: {n} tiradas en "
        f"{resumen.get('capturas', 0)} imágenes."
        if n
        else "Aún no hay tiradas analizadas. Copia tus capturas en imagenes/capturas y pulsa Analizar."
    )
    globals_["filas"] = [
        ["Tiradas analizadas", n, "100%", "—"],
        ["Imágenes procesadas", resumen.get("capturas", 0), "—", "—"],
        ["Números distintos detectados", resumen["numeros_distintos"], None, "37"],
        ["Veces que salió el 0", resumen["ceros"], pct(resumen["ceros"], n), "2.70%"],
        ["Veces rojo", resumen["colores"].get("rojo", 0), pct(resumen["colores"].get("rojo", 0), n), "48.65%"],
        ["Veces negro", resumen["colores"].get("negro", 0), pct(resumen["colores"].get("negro", 0), n), "48.65%"],
        [
            "Racha máxima actual",
            (
                f"{resumen.get('racha_actual', 0)} ({resumen.get('racha_actual_numero')})"
                if resumen.get("racha_actual")
                else 0
            ),
            None,
            "—",
        ],
        ["Racha máxima histórica", resumen["racha_max"], None, "—"],
    ]

    nums = esquema["frecuencia_numeros"]
    for fila in nums["filas"]:
        n0 = fila[0]
        veces = resumen["numeros"].get(n0, 0)
        fila[3] = veces
        fila[4] = pct(veces, n)
        fila[5] = round(esp, 2) if esp else None
        fila[6] = desviacion(veces, esp) if esp else None

    docs = esquema["frecuencia_docenas"]
    for fila in docs["filas"]:
        clave = fila[0]
        veces = resumen["docenas"].get(clave, 0)
        esperado = n / 3.0
        fila[2] = veces
        fila[3] = pct(veces, n)
        fila[4] = round(esperado, 2) if esperado else None
        fila[5] = desviacion(veces, esperado) if esperado else None

    col = esquema["frecuencia_colores"]
    for fila in col["filas"]:
        nombre = str(fila[0]).split()[0]
        veces = resumen["colores"].get(nombre, 0)
        esperado = {"rojo": 0.4865, "negro": 0.4865, "verde": 0.0270}.get(nombre)
        fila[2] = veces
        fila[3] = pct(veces, n)
        fila[5] = desviacion(veces, n * esperado) if (n and esperado) else None

    _rellenar_repeticiones(esquema["repeticion_docenas"], resumen["repeticiones"]["docenas"])
    _rellenar_alternancia(esquema["alternancia_docenas"], resumen["alternancia_docenas"])
    return esquema


def pct(parte, total):
    return round(100.0 * parte / total, 2) if total else None


def desviacion(observado, esperado):
    if not esperado:
        return None
    d = 100.0 * (observado - esperado) / esperado
    return f"{d:+.1f}%"


def _rellenar_repeticiones(tabla: dict, datos: dict) -> None:
    total_global = 0
    for fila in tabla["filas"]:
        clave = fila[0]
        # la etiqueta es texto; se busca por coincidencia de prefijo
        d = _buscar_datos(datos, clave)
        if d is None:
            continue
        for i, campo in enumerate(tabla["campos"]):
            fila[1 + i] = d.get(campo, 0)
        fila[1 + len(tabla["campos"])] = d.get("Total", 0)
        total_global += d.get("Total", 0)
    for fila in tabla["filas"]:
        d = _buscar_datos(datos, fila[0])
        if d is None:
            continue
        fila[-1] = pct(d.get("Total", 0), total_global)
    tot = tabla["totales"]
    for i, campo in enumerate(tabla["campos"]):
        tot[1 + i] = sum(d.get(campo, 0) for d in datos.values())
    tot[1 + len(tabla["campos"])] = total_global
    tot[-1] = "100%" if total_global else None


def _buscar_datos(datos: dict, etiqueta: str):
    """Las etiquetas son texto ('2 columna (1, 4, ...)'); las claves, '2' o 1."""
    prefijo = etiqueta.split()[0].rstrip(".")
    if prefijo in datos:
        return datos[prefijo]
    try:
        clave = int(prefijo)
    except ValueError:
        return None
    return datos.get(clave)


def _rellenar_alternancia(tabla: dict, alt: dict) -> None:
    mapa = {
        "Tiradas con docena": alt["tiradas_con_docena"],
        "Pares adyacentes comparados": alt["pares_adyacentes"],
        "Cambios de docena": alt["cambios"],
        "% de alternancia": alt["pct_alternancia"],
        "Tiradas por cada cambio": alt["tiradas_por_cambio"],
        "Repeticiones iguales seguidas": alt["repeticiones_iguales"],
        "Racha media (misma docena)": alt["racha_media"],
        "Racha máxima (misma docena)": alt["racha_max"],
    }
    for fila in tabla["filas"]:
        v = mapa.get(fila[0])
        fila[1] = f"{v}%" if fila[0] == "% de alternancia" and v is not None else v
    for fila in tabla["filas_docena"]:
        d = alt["por_docena"].get(fila[0])
        if d:
            fila[1] = d["apariciones"]
            fila[2] = d["tiradas_entre_apariciones"]
            fila[3] = d["max_tiradas_ausente"]
