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
from core import simulador_escalonado as ESC
from core import vision
from core.simulador import cop

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_CAPTURAS = RAIZ / "imagenes" / "capturas"
EXTENSIONES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Una subcarpeta por casino. La raíz de `capturas` guarda las imágenes viejas.
CARPETAS_CASINO = ["betplay", "rushbet", "wplay", "melbet"]


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ruta_carpeta(nombre: str) -> Path:
    """Carpeta de un casino. `nombre` vacío significa la raíz de capturas."""
    if not nombre:
        return CARPETA_CAPTURAS
    return CARPETA_CAPTURAS / nombre


class Analizador:
    """Mantiene un único análisis en curso y su progreso."""

    def __init__(self) -> None:
        # RLock, no Lock: iniciar() escribe en el log desde dentro del candado
        self._candado = threading.RLock()
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
    def hay_capturas(self, carpeta: str = "") -> bool:
        return any(
            p.suffix.lower() in EXTENSIONES for p in ruta_carpeta(carpeta).glob("*")
        )

    def contar_capturas(self, carpeta: str = "") -> int:
        return sum(
            1 for p in ruta_carpeta(carpeta).glob("*") if p.suffix.lower() in EXTENSIONES
        )

    def limpiar_ausentes(self) -> int:
        """Quita del historial las capturas cuyo archivo ya no está en la carpeta.

        Se llama antes de comprobar si hay imágenes, para que al borrar todo
        el dashboard no se quede con los datos de las capturas anteriores.
        """
        return BD.podar_capturas_inexistentes(CARPETA_CAPTURAS)

    def iniciar(self, reiniciar_historial: bool = False, carpeta: str = "") -> bool:
        if carpeta and carpeta not in CARPETAS_CASINO:
            return False
        with self._candado:
            if self._hilo and self._hilo.is_alive():
                return False
            # lo que ya no está en disco no debe seguir contando como captura
            podadas = self.limpiar_ausentes()
            if podadas:
                self._log(f"Limpieza: {podadas} captura(s) borrada(s) del disco, "
                          "se quitaron del contador")
            etiqueta = carpeta or "capturas"
            self.estado = {
                "activo": True,
                "carpeta": carpeta,
                "total": self.contar_capturas(carpeta),
                "procesadas": 0,
                "actual": None,
                "log": [f"Analizando la carpeta: {etiqueta}"],
                "iniciado_en": _ahora(),
                "terminado_en": None,
                "errores": 0,
                "tiradas_nuevas": 0,
            }
            hilo = threading.Thread(
                target=self._trabajar,
                args=(reiniciar_historial, carpeta),
                daemon=True,
            )
            self._hilo = hilo
        hilo.start()
        return True

    # ------------------------------------------------------------- worker
    def _trabajar(self, reiniciar: bool, carpeta: str = "") -> None:
        try:
            if reiniciar:
                # solo se borra el historial del casino elegido, no el de los demás
                BD.limpiar(carpeta)
                self._log("Historial anterior borrado.")

            cfg = vision.cargar_config()
            rutas = sorted(
                p
                for p in ruta_carpeta(carpeta).glob("*")
                if p.suffix.lower() in EXTENSIONES
            )
            inicio = time.time()
            nuevas = 0

            for ruta in rutas:
                with self._candado:
                    self.estado["actual"] = ruta.name
                self._analizar_una(ruta, cfg, carpeta)
                with self._candado:
                    self.estado["procesadas"] += 1
                time.sleep(0.0)  # cede el hilo para que la UI refresque

            nuevas = BD.resumen_capturas(carpeta or None)["tiradas"]
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
            # una foto más de la comparativa: el casino queda registrado con
            # el neto que tendría la escalada sobre todo su historial
            if carpeta:
                sim = registrar_resultado_casino(carpeta)
                if sim:
                    self._log(
                        f"{carpeta}: escalada sobre {sim['tiradas']} tiradas → "
                        f"neto {cop(sim['neto'])} COP"
                    )
        except Exception:
            self._log("Fallo grave:\n" + traceback.format_exc())
            with self._candado:
                self.estado["activo"] = False
                self.estado["terminado_en"] = _ahora()

    def _analizar_una(self, ruta: Path, cfg: dict, carpeta: str = "") -> None:
        # reprocesar una captura reemplaza sus filas anteriores en vez de duplicarlas
        BD.borrar_por_archivo(ruta.name, carpeta)
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
                carpeta=carpeta,
            )
            return

        umbral_alto = float(cfg.get("umbral_confianza_alta", 0.70))
        umbral_color = float(cfg.get("umbral_confianza_con_color", 0.55))
        filas = []
        dudosas = 0
        for celda in res.celdas:
            if celda.get("numero") is None:
                continue
            conf = float(celda.get("confianza") or 0.0)
            # El color de la ruleta es una verdad conocida: si el dígito leído
            # concuerda con el color de su casilla, la lectura es de fiar aunque
            # la confianza sea baja. Medido sobre las capturas reales, esto sube
            # la precisión del 73% al 90% sin tirar las lecturas buenas.
            color_ok = bool(celda.get("color_coincide"))
            if conf >= umbral_alto or (conf >= umbral_color and color_ok):
                estado = "ok"
            else:
                estado = "duda"
                dudosas += 1
            filas.append(
                (
                    int(celda["numero"]),
                    conf,
                    celda.get("color_esperado") or "",
                    estado,
                )
            )

        BD.registrar_captura(
            archivo=ruta.name,
            ruta=str(ruta),
            analizada_en=_ahora(),
            modo=res.modo,
            carpeta=carpeta,
            n_detectadas=res.n_detectadas,
            n_reconocidas=res.n_reconocidas,
            n_dudas=dudosas,
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
def metricas_completas(limite_capturas: int = 50, carpeta: str = "") -> dict:
    """Todo lo que el dashboard necesita, en una sola respuesta.

    `carpeta` filtra por casino: con `""` se mezclan todos.
    """
    filtro = carpeta or None
    tiradas = BD.todas_las_tiradas(filtro)
    historial = BD.resumen_capturas(filtro)
    resumen = E.resumen_desde_tiradas(tiradas)
    resumen["capturas"] = historial["capturas_ok"]
    esquema = E.esquema_tablas("docenas")
    return {
        "esquema": esquema,
        "resumen": resumen,
        "capturas": historial,
        "carpeta": carpeta,
        "capturas_disponibles": ANALIZADOR.contar_capturas(carpeta),
        "carpetas_disponibles": {
            c: ANALIZADOR.contar_capturas(c) for c in CARPETAS_CASINO
        },
        "ultimas_capturas": BD.ultimas_capturas(limite_capturas, filtro),
        "ajustes_rejilla": BD.listar_ajustes(),
        "alternancia": resumen["alternancia_docenas"],
        "progreso": ANALIZADOR.snapshot(),
        "sin_datos": not tiradas,
    }


# ------------------------------------------------------------------ comparativa
def registrar_resultado_casino(casino: str, base: int = 2500) -> dict | None:
    """Simula la escalada sobre todo el historial de un casino y lo archiva.

    Se llama al terminar cada análisis, así la tabla de la comparativa va
    guardando la foto de cómo estaba cada casino cada vez que se procesa.
    """
    tiradas = BD.todas_las_tiradas(casino)
    if not tiradas:
        return None
    sim = ESC.simular(tiradas, base, True)
    hist = BD.resumen_capturas(casino)
    BD.guardar_resultado(casino, sim, hist["capturas_ok"], hist["dudas"])
    return sim


def comparativa(base: int = 2500, incluir_cero: bool = True) -> dict:
    """Corre la escalada sobre el historial completo de los 4 casinos.

    No es una predicción: es lo que habría dado la estrategia sobre las
    capturas que llevas analysing. Cada casino se simula por separado y luego
    se ordenan de mejor a peor por dinero neto.
    """
    filas = []
    for casino in CARPETAS_CASINO:
        tiradas = BD.todas_las_tiradas(casino)
        hist = BD.resumen_capturas(casino)
        sim = ESC.simular(tiradas, base, incluir_cero) if tiradas else None
        filas.append(
            {
                "casino": casino,
                "capturas": hist["capturas_ok"],
                "dudas": hist["dudas"],
                "tiradas": len(tiradas),
                "simulacion": sim,
                "neto": int(sim["neto"]) if sim else 0,
                "apostado": int(sim["total_apostado"]) if sim else 0,
                "devuelto": int(sim["total_devuelto"]) if sim else 0,
                "apuestas": int(sim["apuestas"]) if sim else 0,
                "aciertos": int(sim["aciertos"]) if sim else 0,
                "tasa_acerto": sim["tasa_acerto"] if sim else None,
                "roi_pct": sim["roi_pct"] if sim else None,
                "racha_max": sim["impactos"]["tras_dos_golpes"]["apuestas"] if sim else 0,
                "sin_datos": not tiradas,
            }
        )

    # mejor por dinero neto, entre los que tienen historial
    con_datos = [f for f in filas if not f["sin_datos"]]
    con_datos.sort(key=lambda f: f["neto"], reverse=True)
    ganador = con_datos[0] if con_datos else None

    totales = {
        "capturas": sum(f["capturas"] for f in filas),
        "tiradas": sum(f["tiradas"] for f in filas),
        "apuestas": sum(f["apuestas"] for f in filas),
        "apostado": sum(f["apostado"] for f in filas),
        "devuelto": sum(f["devuelto"] for f in filas),
        "neto": sum(f["neto"] for f in filas),
    }
    totales["roi_pct"] = (
        round(100.0 * totales["neto"] / totales["apostado"], 2)
        if totales["apostado"]
        else None
    )

    return {
        "base": base,
        "apuesta_triple": base * 3,
        "incluye_cero": incluir_cero,
        "filas": sorted(filas, key=lambda f: f["neto"], reverse=True),
        "ganador": ganador,
        "peor": con_datos[-1] if con_datos else None,
        "totales": totales,
        "casinos_con_datos": len(con_datos),
        "regla": ESC.simular([0, 1, 2, 3], base, incluir_cero)["regla"],
        "historial": BD.historial_resultados(120),
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
