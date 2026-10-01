from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from core import basedatos as BD  # noqa: E402
from core import estadisticas as E  # noqa: E402
from core import orquestador as O  # noqa: E402
from core import ruleta_reglas as R  # noqa: E402
from core import simulador as SIM  # noqa: E402
from core import simulador_escalonado as ESC  # noqa: E402
from core import vision  # noqa: E402

IMAGENES_DIR = BASE_DIR / "imagenes"
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"

app = FastAPI(title="Analizador de Ruleta", version="0.2.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SUBCARPETAS_IMAGEN = ["capturas", "recortes", "dataset"]
CASINOS = O.CARPETAS_CASINO  # betplay, rushbet, wplay, melbet


def _validar_casino(nombre: str) -> str:
    """Devuelve el nombre de casino válido, o cadena vacía (raíz)."""
    if not nombre:
        return ""
    if nombre not in CASINOS:
        raise HTTPException(
            status_code=400,
            detail=f"Casino desconocido: {nombre}. Usa uno de {CASINOS}.",
        )
    return nombre


@app.on_event("startup")
def _arrancar() -> None:
    BD.inicializar()


def _contar_imagenes() -> dict:
    conteo = {}
    total = 0
    for sub in SUBCARPETAS_IMAGEN:
        carpeta = IMAGENES_DIR / sub
        n = (
            sum(1 for p in carpeta.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONES_IMAGEN)
            if carpeta.exists()
            else 0
        )
        conteo[sub] = n
        total += n
    conteo["raiz"] = sum(
        1
        for p in IMAGENES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONES_IMAGEN
    )
    conteo["total"] = total + conteo["raiz"]
    return conteo


# ------------------------------------------------------------------ páginas
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    # El panel de reglas de la ruleta está oculto: solo queda la línea de
    # contexto (casillas, edge y RTP) en el encabezado.
    totales = R.resumen()
    # las tablas se rellenan con datos reales: si no, el Jinja deja guiones
    casino = request.query_params.get("casino", "")
    metricas = O.metricas_completas(carpeta=_validar_casino(casino))
    O.rellenar_tabla(metricas["esquema"], metricas["resumen"])
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "tablas": metricas["esquema"],
            "campos_repeticion": E.CAMPOS_REPETICION,
            "agrupaciones": E.AGRUPACIONES,
            "imagenes": _contar_imagenes(),
            "casinos": CASINOS,
            "casino_sel": _validar_casino(casino) or CASINOS[0],
            "contexto": {"totales": totales},
            "payload_simulador": SIM.simular([], 2500)["regla"],
        "payload_escalonado": ESC.simular([], 2500)["regla"],
        },
    )


# ---------------------------------------------------------------------- API
@app.get("/api/esquema")
def api_esquema(casino: str = ""):
    datos = O.metricas_completas(carpeta=_validar_casino(casino))
    O.rellenar_tabla(datos["esquema"], datos["resumen"])
    return datos


@app.get("/api/reglas")
def api_reglas():
    return R.resumen()


@app.get("/api/imagenes")
def api_imagenes(casino: str = ""):
    casino = _validar_casino(casino)
    carpeta = O.ruta_carpeta(casino)
    archivos = sorted(
        p.name for p in carpeta.glob("*") if p.suffix.lower() in O.EXTENSIONES
    )
    return {
        "carpetas": _contar_imagenes(),
        "casinos": CASINOS,
        "casino": casino,
        "raiz": str(IMAGENES_DIR),
        "ruta_carpeta": str(carpeta),
        "capturas_disponibles": len(archivos),
        "por_casino": {c: O.ANALIZADOR.contar_capturas(c) for c in CASINOS},
        "archivos": archivos,
    }


class PeticionAnalisis(BaseModel):
    reiniciar_historial: bool = False
    casino: str = ""


@app.post("/api/analizar")
def api_analizar(peticion: PeticionAnalisis):
    casino = _validar_casino(peticion.casino)
    # primero se va lo que ya no está en disco, aunque la carpeta quede vacía
    O.ANALIZADOR.limpiar_ausentes()
    if not O.ANALIZADOR.hay_capturas(casino):
        raise HTTPException(
            status_code=400,
            detail=(
                f"No hay imágenes en '{casino or 'capturas'}'. Copia tus capturas en "
                f"{O.ruta_carpeta(casino)} y vuelve a intentarlo."
            ),
        )
    if not O.ANALIZADOR.iniciar(peticion.reiniciar_historial, casino):
        raise HTTPException(status_code=409, detail="Ya hay un análisis en curso.")
    return {"iniciado": True, "estado": O.ANALIZADOR.snapshot()}


@app.get("/api/estado")
def api_estado():
    return O.ANALIZADOR.snapshot()


@app.get("/api/resultados")
def api_resultados(limite: int = 50, casino: str = ""):
    filtro = _validar_casino(casino) or None
    return {
        "ultimas_capturas": BD.ultimas_capturas(max(1, min(limite, 200)), filtro),
        "resumen": BD.resumen_capturas(filtro),
    }


@app.get("/api/numeros")
def api_numeros(limite: int = 500, casino: str = ""):
    """Lista exacta de los números leídos, para verificar el OCR a mano."""
    carpeta = _validar_casino(casino) or None
    secuencia = BD.secuencia_tiradas(max(1, min(limite, 2000)), carpeta)
    # Las dudosas NO vienen en `secuencia` (allí solo entran las 'ok'), así que
    # hay que pedirlas aparte. Antes se filtraba la lista buena, que por
    # definición ya no contenía ninguna, y el contador marcaba siempre 0.
    dudosas = BD.tiradas_dudosas(max(1, min(limite, 2000)), carpeta)
    return {
        "secuencia": secuencia,
        "numeros": [f["numero"] for f in secuencia],
        "total": len(secuencia),
        "dudosos": dudosas,
        "total_dudosas": len(dudosas),
        "confianza_media": round(
            sum(f["confianza"] for f in secuencia) / len(secuencia), 3
        )
        if secuencia
        else None,
    }


class PeticionSimulacion(BaseModel):
    apuesta: int = Field(default=2500, gt=0, le=100_000_000)
    incluir_cero: bool = True
    usar_analizadas: bool = True
    tiradas: list[int] | None = None
    disparo: int = 2
    racha_max: int | None = None
    casino: str = ""


@app.post("/api/simulador")
def api_simulador(peticion: PeticionSimulacion):
    if peticion.usar_analizadas or not peticion.tiradas:
        tiradas = BD.todas_las_tiradas(_validar_casino(peticion.casino) or None)
        origen = "tiradas leídas de las imágenes"
    else:
        tiradas = [t for t in peticion.tiradas if isinstance(t, int) and 0 <= t <= 36]
        origen = "secuencia manual"
    if not tiradas:
        raise HTTPException(
            status_code=400,
            detail="No hay tiradas. Analiza capturas o envía una secuencia manual.",
        )
    return {
        "origen": origen,
        "tiradas": len(tiradas),
        **SIM.simular(
            tiradas,
            peticion.apuesta,
            peticion.incluir_cero,
            peticion.disparo,
            peticion.racha_max,
        ),
    }


class PeticionEscalonado(BaseModel):
    base: int = 2500
    incluir_cero: bool = True
    usar_analizadas: bool = True
    tiradas: list[int] | None = None
    casino: str = ""


@app.post("/api/escalonado")
def api_escalonado(peticion: PeticionEscalonado):
    """Estrategia de escalada: 2.500 al segundo golpe, 7.500 al tercero."""
    if peticion.usar_analizadas or not peticion.tiradas:
        tiradas = BD.todas_las_tiradas(_validar_casino(peticion.casino) or None)
        origen = "tiradas leídas de las imágenes"
    else:
        tiradas = [t for t in peticion.tiradas if isinstance(t, int) and 0 <= t <= 36]
        origen = "secuencia manual"
    if not tiradas:
        raise HTTPException(
            status_code=400,
            detail="No hay tiradas. Analiza capturas o envía una secuencia manual.",
        )
    return {
        "origen": origen,
        **ESC.simular(tiradas, peticion.base, peticion.incluir_cero),
    }


@app.get("/api/comparativa")
def api_comparativa(base: int = 2500, incluir_cero: bool = True):
    """Compara los 4 casinos con la estrategia de escalada sobre su historial."""
    return O.comparativa(max(100, min(base, 100000)), incluir_cero)


@app.delete("/api/historial")
def api_limpiar_historial(casino: str = ""):
    nombre = _validar_casino(casino)
    BD.limpiar(nombre or None)
    return {"historial": "borrado", "casino": nombre}


# ------------------------------------------------------------- calibración
class Rejilla(BaseModel):
    x: float
    y: float
    paso_x: float = Field(gt=0)
    paso_y: float = Field(gt=0)
    columnas: int = Field(gt=0)
    filas: int = Field(gt=0)


class PeticionAjuste(BaseModel):
    nombre: str = Field(min_length=1, max_length=80)
    rejilla: Rejilla


@app.get("/api/calibracion")
def api_calibracion_listar():
    return {"ajustes": BD.listar_ajustes()}


@app.post("/api/calibracion")
def api_calibracion_guardar(peticion: PeticionAjuste):
    BD.guardar_ajuste(peticion.nombre, peticion.rejilla.model_dump())
    return {"guardado": peticion.nombre, "ajustes": BD.listar_ajustes()}


@app.delete("/api/calibracion/{nombre}")
def api_calibracion_borrar(nombre: str):
    with BD._candado, BD.conexion() as con:
        con.execute("DELETE FROM ajustes_rejilla WHERE nombre = ?", (nombre,))
    return {"borrado": nombre}


# ------------------------------------------------------ diagnóstico puntual
@app.get("/api/diagnostico")
def api_diagnostico(archivo: str, casino: str = ""):
    ruta = O.ruta_carpeta(_validar_casino(casino)) / Path(archivo).name
    if not ruta.exists() or ruta.suffix.lower() not in O.EXTENSIONES:
        raise HTTPException(status_code=404, detail="Captura no encontrada.")
    cfg = vision.cargar_config()
    res = vision.analizar_imagen(ruta, cfg)
    necesita = bool(res.necesita_calibracion(cfg))
    return JSONResponse(
        {
            "archivo": ruta.name,
            **res.a_json(),
            "necesita_calibracion": necesita,
            "sugerencia": (
                "Ajusta la rejilla y guárdala con un nombre para este tipo de captura."
                if necesita
                else "Detección fiable: no hace falta calibrar."
            ),
        }
    )
