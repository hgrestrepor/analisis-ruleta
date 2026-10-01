from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

from core import plantillas as P
from core import ruleta_reglas as R

RAIZ = Path(__file__).resolve().parents[2]
RUTA_CONFIG = RAIZ / "config" / "deteccion.json"

CONFIG_POR_DEFECTO = {
    "modo": "auto",
    "invertir_orden": False,
    "escala": 1.0,
    "umbral_confianza": 0.62,
    "umbral_tinta": 26,
    "hueco_max_digitos": None,
    "factor_fusion": 0.30,
    "tolerancia_rejilla": 0.35,
    "percentil_rejilla": 0.85,
    "kernel_por_altura": 0.22,
    "hueco_por_altura": 0.30,
    "umbral_minimo": 6.0,
    "ancho_simple_max": 1.35,
    "minimo_celdas_rejilla": 4,
    "roi": None,
    "rejilla": None,
    "color": {
        "s_min": 60,
        "v_min": 40,
        "rojo_h": [[0, 16], [164, 179]],
        "verde_h": [[30, 90]],
        "negro_v_max": 82,
        "negro_s_max": 120,
    },
}


def cargar_config() -> dict:
    cfg = json.loads(json.dumps(CONFIG_POR_DEFECTO))
    if RUTA_CONFIG.is_file():
        usuario = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))
        for clave, valor in usuario.items():
            if isinstance(valor, dict) and isinstance(cfg.get(clave), dict):
                cfg[clave].update(valor)
            else:
                cfg[clave] = valor
    return cfg


@dataclass
class Celda:
    numero: int | None
    color: str
    confianza: float
    digitos: str
    x: int
    y: int
    w: int
    h: int
    fila: int = 0
    color_coincide: bool = True
    alternativa: int | None = None
    reparado: bool = False
    # multipase: cuántos pases vieron este dígito y en cuáles discrepan
    votos: int = 1
    pases_totales: int = 1
    desacuerdos: list[int] = field(default_factory=list)

    def a_dict(self) -> dict:
        return asdict(self)

    @property
    def fiable(self) -> bool:
        """Todos los pases que vieron la celda coincidieron."""
        return self.votos == self.pases_totales and not self.desacuerdos


@dataclass
class ResultadoImagen:
    archivo: str
    ruta: str
    modo: str
    celdas: list[dict] = field(default_factory=list)
    tiradas: list[int] = field(default_factory=list)
    confianza_promedio: float = 0.0
    n_detectadas: int = 0
    n_reconocidas: int = 0
    n_dudas: int = 0
    n_conflicto_color: int = 0
    errores: list[str] = field(default_factory=list)
    advertencia: str | None = None
    rejilla: dict | None = None
    multipase: dict | None = None
    tabla: dict | None = None

    def necesita_calibracion(self, cfg: dict) -> bool:
        """Indica si conviene pedir rejilla manual antes de confiar el resultado."""
        if self.modo == "cuadros":
            # las celdas salieron de las reglas y del perfil de tinta de la propia
            # captura, no de un ajuste de retícula: el residuo del ajuste no dice
            # nada de esta lectura
            return False
        if self.rejilla is None:
            return True
        return bool(
            float(self.rejilla.get("residuo_rel", 0.0))
            > float(cfg.get("residuo_calibracion", 0.45))
        )

    def a_json(self) -> dict:
        """Vista serializable del resultado (convierte tipos de NumPy)."""
        return {
            "archivo": self.archivo,
            "modo": self.modo,
            "tiradas": self.tiradas,
            "confianza_promedio": round(float(self.confianza_promedio), 4),
            "n_detectadas": int(self.n_detectadas),
            "n_reconocidas": int(self.n_reconocidas),
            "n_dudas": int(self.n_dudas),
            "n_conflicto_color": int(self.n_conflicto_color),
            "errores": list(self.errores),
            "advertencia": self.advertencia,
            "rejilla": self.rejilla,
            "multipase": self.multipase,
            "tabla": self.tabla,
        }


def cargar_imagen(ruta: Path) -> np.ndarray | None:
    try:
        datos = np.fromfile(str(ruta), dtype=np.uint8)
    except OSError:
        return None
    if datos.size == 0:
        return None
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if imagen is None or imagen.size == 0:
        return None
    if imagen.ndim == 2:
        return cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    return imagen


def aplicar_roi(imagen: np.ndarray, roi) -> np.ndarray:
    if not roi:
        return imagen
    x, y, w, h = (int(v) for v in roi)
    h_img, w_img = imagen.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x0 + w), min(h_img, y0 + h)
    if x1 <= x0 or y1 <= y0:
        return imagen
    return imagen[y0:y1, x0:x1]


def _color_fondo(imagen: np.ndarray) -> np.ndarray:
    h, w = imagen.shape[:2]
    grosor = max(2, min(h, w) // 60)
    bordes = np.concatenate(
        [
            imagen[:grosor].reshape(-1, 3),
            imagen[-grosor:].reshape(-1, 3),
            imagen[:, :grosor].reshape(-1, 3),
            imagen[:, -grosor:].reshape(-1, 3),
        ]
    )
    return np.median(bordes, axis=0)


def _mascara_cuerpos(imagen: np.ndarray, cfg: dict) -> np.ndarray:
    fondo = _color_fondo(imagen).astype(np.float32)
    dist = np.linalg.norm(imagen.astype(np.float32) - fondo, axis=2)
    return (dist > float(cfg["distancia_fondo"])).astype(np.uint8) * 255


def _mascara_digitos_oscuros(celda: np.ndarray) -> np.ndarray:
    gris = cv2.cvtColor(celda, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_osc = int((binaria == 0).sum())
    n_claro = int((binaria == 255).sum())
    if n_osc == 0 or n_claro == 0:
        return binaria
    minoritaria = binaria == 0 if n_osc <= n_claro else binaria == 255
    return minoritaria.astype(np.uint8) * 255


def _componentes(mascara: np.ndarray, area_min: float) -> list[tuple[int, int, int, int]]:
    num, _, stats, _ = cv2.connectedComponentsWithStats(mascara, 8)
    salida = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < area_min or w < 3 or h < 6:
            continue
        if w / float(h) > 3.6:
            continue
        salida.append((x, y, w, h))
    return salida


def altura_minima_digito(imagen: np.ndarray) -> float:
    """Altura mínima creíble para un glifo de historial.

    Sirve para distinguir "detecté puntuación" de "detecté cifras": una coma
    o un separador miden una fracción del alto de la captura, mientras que
    un número de ruleta siempre ocupa un tamaño apreciable del área.
    """
    return max(6.0, 0.030 * min(imagen.shape[0], imagen.shape[1]))


def _mascara_absdiff(gris: np.ndarray, k: int, umbral: float) -> np.ndarray:
    if k % 2 == 0:
        k += 1
    fondo_local = cv2.blur(gris, (k, k))
    diff = cv2.absdiff(gris, fondo_local)
    return (diff > umbral).astype(np.uint8) * 255


def _mascara_tinta(imagen: np.ndarray, cfg: dict) -> np.ndarray:
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    base = float(cfg["umbral_tinta"])
    piso = float(cfg.get("umbral_minimo", 6.0))
    k = int(max(5, min(41, imagen.shape[0] // 20)))
    proporcion = float(cfg.get("kernel_por_altura", 0.22))

    mejor: tuple[int, int, np.ndarray, int] | None = None
    for indice, umbral in enumerate((base, max(piso, base * 0.5), max(piso, base * 0.22))):
        k_actual = k
        for _ in range(4):
            mascara = _mascara_absdiff(gris, k_actual, umbral)
            blobs = _blobs_digitos(mascara)
            if not blobs:
                break
            alturas = sorted(b[3] for b in blobs)
            mediana = alturas[len(alturas) // 2]
            k_nuevo = int(max(3, min(41, round(proporcion * mediana))))
            estable = abs(k_nuevo - k_actual) <= 1
            if mejor is None or len(blobs) > mejor[0]:
                mejor = (len(blobs), k_actual, mascara, indice)
            if estable:
                break
            k_actual = k_nuevo
        # Solo se afloja el umbral si lo detectado es demasiado pequeño para
        # ser una cifra (puntuación, comas, viñetas). Un umbral bajo en una
        # imagen normal solo añadiría ruido, así que no se prueba.
        if mejor is not None:
            candidatas = _blobs_digitos(mejor[2])
            if candidatas:
                alturas = sorted(b[3] for b in candidatas)
                mediana = alturas[len(alturas) // 2]
                if mediana >= altura_minima_digito(imagen):
                    break

    if mejor is None:
        return np.zeros(gris.shape, np.uint8)
    return _quitar_lineas(mejor[2])


def _cuenta_reglas(mascara: np.ndarray, minimo: int, grosor_max: int = 5) -> int:
    """Número de filas con un trazo de ancho casi completo y delgado."""
    tinta = (mascara > 0).astype(np.uint8)
    h, w = tinta.shape
    total = 0
    for y in range(h):
        fila = tinta[y]
        if not fila.any():
            continue
        bordes = np.flatnonzero(np.diff(np.concatenate(([0], fila, [0]))))
        for inicio, fin in zip(bordes[0::2], bordes[1::2]):
            if fin - inicio < minimo:
                continue
            grosor = 0
            for y2 in range(y, min(h, y + grosor_max + 3)):
                if tinta[y2, inicio:fin].mean() >= 0.8:
                    grosor += 1
                else:
                    break
            if 0 < grosor <= grosor_max:
                total += 1
                break
    return total


def _borrar_reglas(mascara: np.ndarray, minimo: int, grosor_max: int = 5) -> np.ndarray:
    """Borra los trazos anchos y delgados de una orientación."""
    tinta = (mascara > 0).astype(np.uint8)
    h, w = tinta.shape
    salida = mascara.copy()
    for y in range(h):
        fila = tinta[y]
        if not fila.any():
            continue
        bordes = np.flatnonzero(np.diff(np.concatenate(([0], fila, [0]))))
        for inicio, fin in zip(bordes[0::2], bordes[1::2]):
            if fin - inicio < minimo:
                continue
            grosor = 0
            for y2 in range(y, min(h, y + grosor_max + 3)):
                if tinta[y2, inicio:fin].mean() >= 0.8:
                    grosor += 1
                else:
                    break
            if 0 < grosor <= grosor_max:
                salida[y:y + grosor, inicio:fin] = 0
    return salida


def _quitar_lineas(mascara: np.ndarray) -> np.ndarray:
    """Borra reglas horizontales y verticales de tabla.

    En una captura mal recortada los separadores de la tabla tocan los dígitos
    de arriba y de abajo, y la detección de componentes los fusiona en un blob
    gigante. Ese blob no parece un número y se descarta entero, así que se
    perdían dígitos de golpe.

    Solo se aplica si la imagen es de verdad una tabla, es decir, si tiene
    varios trazos que cruzan el ancho completo. Sin esa condición el borrado
    destrozaba los dígitos: en un historial compacto los de una fila van pegados
    y forman un trazo continuo igual de largo, y los bordes superiores de las
    cifras son trazos delgados. Se comprobó con la captura de una sola fila,
    que se quedaba a 0 de 20.
    """
    if mascara.size == 0:
        return mascara
    h, w = mascara.shape[:2]
    completas = int(w * 0.85)
    if _cuenta_reglas(mascara, completas) < 3:
        return mascara
    horizontal = _borrar_reglas(mascara, max(20, int(w * 0.45)))
    vertical = np.ascontiguousarray(
        _borrar_reglas(np.ascontiguousarray(horizontal.T), max(20, int(h * 0.45))).T
    )
    return vertical


def _blobs_digitos(mascara: np.ndarray) -> list[tuple[int, int, int, int]]:
    area_total = mascara.size
    candidatos = _componentes(mascara, area_total * 0.00002)
    if not candidatos:
        return []
    alturas = sorted(c[3] for c in candidatos)
    mediana = alturas[len(alturas) // 2]
    if mediana <= 0:
        return []
    # Sin techo de ancho se colaban textos y bordes de la interfaz: en una captura
    # mal recortada los rótulos y las reglas de la tabla llegaban a medir 100 px de
    # ancho con 20 de alto, contaminando el ajuste de la retícula. Una cifra, o un
    # número de dos cifras, nunca es tan ancha respecto a su propia altura.
    ancho_max = max(4.0 * mediana, mediana)
    return [
        (x, y, w, h)
        for x, y, w, h in candidatos
        if 0.42 * mediana <= h <= 2.4 * mediana
        and 0.18 * mediana <= w <= ancho_max
        and w <= 4.5 * h
    ]


def _umbral_otsu(valores: np.ndarray) -> float:
    if valores.size < 2:
        return float("inf")
    escala = np.clip(valores, 0, 255).astype(np.uint8)
    _t, _bin = cv2.threshold(escala, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(_t)


def _grupos_por_hueco(
    blobs: list[tuple[int, int, int, int]], cfg: dict
) -> list[list[tuple[int, int, int, int]]]:
    if len(blobs) <= 1:
        return [blobs] if blobs else []
    huecos = np.array(
        [b[0] - (a[0] + a[2]) for a, b in zip(blobs, blobs[1:])], dtype=np.float32
    )
    override = cfg.get("hueco_max_digitos")
    if override:
        umbral = float(override)
    else:
        valle = _umbral_valle(huecos)
        if valle is not None:
            umbral = valle
        elif huecos.max() - huecos.min() < 1.0:
            return [[b] for b in blobs]
        else:
            umbral = max(1.0, _umbral_otsu(huecos))
    grupos = [[blobs[0]]]
    for previo, actual, hueco in zip(blobs, blobs[1:], huecos):
        if hueco <= umbral and len(grupos[-1]) < 2:
            grupos[-1].append(actual)
        else:
            grupos.append([actual])
    return grupos

def _filas(blobs: list[tuple[int, int, int, int]], tolerancia: float) -> list[list[tuple[int, int, int, int]]]:
    ordenados = sorted(blobs, key=lambda b: (b[1] + b[3] / 2.0, b[0]))
    filas: list[list[tuple[int, int, int, int]]] = []
    centros: list[float] = []
    for b in ordenados:
        cy = b[1] + b[3] / 2.0
        if filas and abs(cy - centros[-1]) <= tolerancia:
            filas[-1].append(b)
            centros[-1] = sum(x[1] + x[3] / 2.0 for x in filas[-1]) / len(filas[-1])
        else:
            filas.append([b])
            centros.append(cy)
    for f in filas:
        f.sort(key=lambda b: b[0])
    return filas


def _ajustar_pitch(
    centros_por_fila: list[list[float]], escala: float, cfg: dict
) -> tuple[float, float] | None:
    if not centros_por_fila or escala <= 0:
        return None
    tolerancia = float(cfg.get("tolerancia_rejilla", 0.35)) * escala
    minimo = float(cfg.get("minimo_celdas_rejilla", 4))
    percentil = float(cfg.get("percentil_rejilla", 0.85)) * 100.0
    filas = [np.asarray(f, dtype=np.float64) for f in centros_por_fila if len(f)]
    if not filas or max(len(f) for f in filas) < minimo:
        return None
    todos = np.concatenate(filas)
    extension = float(todos.max() - todos.min())
    if extension <= 0:
        return None
    p_max = min(4.0 * escala, extension / 2.0)
    mejor: tuple[float, float, float] | None = None
    for p in np.linspace(1.15 * escala, max(1.15 * escala + 1.0, p_max), 400):
        fases = np.mod(todos, p)
        angulo = 2.0 * np.pi * fases / p
        offset = p * float(
            np.arctan2(np.mean(np.sin(angulo)), np.mean(np.cos(angulo))) / (2.0 * np.pi)
        )
        residuos = []
        for arr in filas:
            idx = np.round((arr - offset) / p)
            residuos.append(np.abs(arr - (offset + idx * p)))
        error = float(np.percentile(np.concatenate(residuos), percentil))
        if error <= tolerancia:
            if mejor is None or p > mejor[0]:
                mejor = (float(p), float(offset), error)
    if mejor is None:
        return None
    return mejor[0], mejor[1]


def _celdas_por_hueco(
    blobs: list[tuple[int, int, int, int]], cfg: dict
) -> list[dict]:
    if not blobs:
        return []
    alturas = sorted(b[3] for b in blobs)
    mediana_h = alturas[len(alturas) // 2]
    limite = float(cfg.get("hueco_por_altura", 0.30)) * mediana_h
    filas = _filas(blobs, tolerancia=0.55 * mediana_h)
    celdas = []
    for indice_fila, fila in enumerate(filas):
        actual: list[tuple[int, int, int, int]] = []
        for b in fila:
            if actual and (b[0] - (actual[-1][0] + actual[-1][2])) > limite:
                caja = _caja_de(actual)
                if caja is not None:
                    caja["fila"] = indice_fila
                    celdas.append(caja)
                actual = []
            actual.append(b)
        if actual:
            caja = _caja_de(actual)
            if caja is not None:
                caja["fila"] = indice_fila
                celdas.append(caja)
    return celdas



def _ajustar_rejilla_2d(
    blobs: list[tuple[int, int, int, int]], cfg: dict
) -> dict | None:
    """Encuentra la retícula real (paso X y paso Y) del historial.

    Las capturas de ruleta son mallas regulares: los chips ocupan posiciones
    fijas. Ajustar esa malla es mucho más fiable que decidir caso por caso si
    dos trazos son un número o dos, porque la retícula impone su propia
    evidencia. Se estimation por media circular de la fase, que es robusta
    aunque alguna celda venga partida en varios blobs.
    """
    if not blobs:
        return None
    alturas = sorted(b[3] for b in blobs)
    mediana_h = alturas[len(alturas) // 2]
    if mediana_h <= 0:
        return None

    centros_x = np.array([b[0] + b[2] / 2.0 for b in blobs], dtype=np.float64)
    cy = np.array([b[1] + b[3] / 2.0 for b in blobs], dtype=np.float64)

    # Ambos ejes se ajustan por mínimos cuadrados sobre toda la captura. Todas
    # las filas comparten las mismas columnas, así que la retícula es global; se
    # estimaba fila por fila y en una fila con 7 u 8 dígitos el rango se repartía
    # en 7 u 8 celdas, saliendo un paso doble del real y la rejilla con 7 columnas
    # en vez de 12, con la mitad de las celdas perdidas.
    #
    # La tolerancia de agrupación sale de la altura del dígito, no del paso: si
    # el paso aproximado fuera un múltiplo del real, agrupando con él se
    # fundirían dos columnas seguidas en una sola.
    ajuste_x = _rejilla_1d(centros_x, 0.85 * mediana_h)
    ajuste_y = _rejilla_1d(cy, 0.85 * mediana_h)
    if ajuste_x is None or ajuste_y is None:
        return None
    x0, paso_x, columnas = ajuste_x
    y0, paso_y, filas = ajuste_y
    if paso_x <= 0 or paso_y <= 0 or columnas < 3 or filas < 2:
        return None

    indices_fila = sorted({int(round((v - y0) / paso_y)) for v in cy if v >= y0})
    if not indices_fila:
        return None

    residuo_x = np.abs(centros_x - (x0 + np.round((centros_x - x0) / paso_x) * paso_x))
    residuo_y = np.abs(cy - (y0 + np.round((cy - y0) / paso_y) * paso_y))
    # El residuo se mide solo sobre los blobs que encajan en la retícula. Los
    # demás son dígitos partidos o desplazados por la máscara, y contarlos hacía
    # que un ajuste bueno pareciera malo (confianza 0) y pidiera calibración.
    dentro = residuo_x <= 0.32 * paso_x
    if int(dentro.sum()) < 3:
        return None
    p90x = float(np.percentile(residuo_x[dentro], 90))
    p90y = float(np.percentile(residuo_y, 90))
    cobertura = float(dentro.sum()) / float(centros_x.size)
    # La confianza combina lo bien que encajan los blobs con cuántos encajan: una
    # retícula sostenida por tres puntos sueltos encaja perfecto y no sirve.
    calidad = max(0.0, 1.0 - p90x / (0.35 * mediana_h))
    return {
        "x": float(x0),
        "y": float(y0),
        "paso_x": float(paso_x),
        "paso_y": float(paso_y),
        "columnas": max(1, int(columnas)),
        "filas": max(1, len(indices_fila)),
        "confianza": round(calidad * cobertura, 4),
        "cobertura": round(cobertura, 4),
        "residuo_x": round(p90x, 2),
        "residuo_y": round(p90y, 2),
        "residuo_rel": round(p90x / mediana_h, 4) if mediana_h else 0.0,
        "y_indices": indices_fila,
    }



def _rejilla_1d(valores: np.ndarray, tolerancia: float) -> tuple[float, float, int] | None:
    """Ajusta una retícula unidimensional sobre las posiciones de un eje.

    Devuelve (origen, paso, número de posiciones). Agrupa los valores con una
    tolerancia tomada de la escala del dígito, y luego resuelve por mínimos
    cuadrados la recta `posicion = origen + i * paso` descartando lo que no
    encaja. Ese descarte es lo que elimina los dígitos que la máscara partió o
    desplazó: si se dejaran, contaban como columnas y filas fantasma y la
    rejilla salía con una posición de más.
    """
    v = np.sort(np.asarray(valores, dtype=np.float64))
    if v.size < 3 or tolerancia <= 0:
        return None
    grupos: list[list[float]] = [[float(v[0])]]
    for x in v[1:]:
        if float(x) - grupos[-1][-1] <= tolerancia:
            grupos[-1].append(float(x))
        else:
            grupos.append([float(x)])
    centros = np.array([float(np.mean(g)) for g in grupos])
    if centros.size < 3:
        return None
    paso = float(np.median(np.diff(centros)))
    x0 = float(centros[0])
    if paso <= 0:
        return None
    indices = np.zeros(0)
    for _ in range(5):
        indices = np.round((centros - x0) / paso)
        dentro = np.abs(centros - (x0 + indices * paso)) <= 0.32 * paso
        if int(dentro.sum()) < 3:
            return None
        i = indices[dentro].astype(np.float64)
        c = centros[dentro]
        n = float(i.size)
        s_i, s_c = float(i.sum()), float(c.sum())
        det = n * float((i * i).sum()) - s_i * s_i
        if abs(det) < 1e-9:
            return None
        paso_nuevo = (n * float((i * c).sum()) - s_i * s_c) / det
        x0_nuevo = (s_c - paso_nuevo * s_i) / n
        if paso_nuevo <= 0:
            return None
        convergio = abs(paso_nuevo - paso) < 0.01 and abs(x0_nuevo - x0) < 0.01
        paso, x0 = paso_nuevo, x0_nuevo
        if convergio:
            break
    return float(x0), float(paso), int(indices.max()) + 1


def _paso_por_autocorrelacion(
    valores: np.ndarray,
    escala: float,
    min_lag: float,
    max_lag: float,
) -> float | None:
    """Periodo dominante de la proyección de densidad de un eje.

    Los chips del historial ocupan posiciones fijas, así que la proyección de
    densidad es periódica y su autocorrelación tiene un pico en el paso
    verdadero. Los picos siguientes son múltiplos: quedarse con uno de ellos
    fusiona columnas de dos en dos y la rejilla sale con la mitad de columnas.
    Por eso se devuelve el primer pico casi tan alto como el mejor, no el mayor,
    y la búsqueda se acota a un rango en torno a la escala del dígito: si el
    rango es abierto, un armónico de 136 px gana a un periodo real de 45 px.
    """
    v = np.asarray(valores, dtype=np.float64)
    if v.size < 3 or escala <= 0 or max_lag <= min_lag:
        return None
    ancho = max(1.0, 0.22 * escala)
    lo, hi = float(v.min()), float(v.max())
    nbins = int((hi - lo) / ancho) + 3
    if nbins < 6 or nbins > 4000:
        return None
    hist, _ = np.histogram(v, bins=nbins, range=(lo - ancho, hi + 2.0 * ancho))
    h = hist.astype(np.float64) - hist.mean()
    ac = np.correlate(h, h, "full")[len(h) - 1:]
    if ac[0] <= 0:
        return None
    lo_lag = max(2, int(min_lag / ancho))
    hi_lag = min(len(ac) - 1, int(max_lag / ancho))
    if hi_lag <= lo_lag:
        return None
    zona = ac[lo_lag:hi_lag + 1]
    if zona.max() <= 0:
        return None
    umbral = 0.9 * float(zona.max())
    for i, valor in enumerate(zona):
        if valor >= umbral:
            return (lo_lag + i) * ancho
    return None



def _celdas_por_rejilla_inferida(
    imagen: np.ndarray,
    blobs: list[tuple[int, int, int, int]],
    cfg: dict,
    ajuste: dict | None = None,
) -> list[dict]:
    """Agrupa los blobs en la retícula ajustada y recorta cada celda completa.

    Los dígitos que la máscara partió se reagrupan solos porque caen en la
    misma celda, y una celda que quedó fusionada con la vecina se recorta al
    ancho del paso, lo que deshace la unión.
    """
    if not blobs:
        return []
    alturas = sorted(b[3] for b in blobs)
    mediana_h = alturas[len(alturas) // 2]
    if mediana_h <= 0:
        return []
    if ajuste is None:
        ajuste = _ajustar_rejilla_2d(blobs, cfg)
    if ajuste is None:
        return []
    paso_x = ajuste["paso_x"]
    paso_y = ajuste["paso_y"]
    x0 = ajuste["x"]
    y0 = ajuste["y"]
    margen = float(cfg.get("margen_celda", 0.06))

    # el paso vertical ajustado puede caer a la mitad del real si las filas
    # alternas están vacías; la altura de celda se toma del espaciado ocupado
    filas_ocupadas = list(ajuste["y_indices"])
    if len(filas_ocupadas) >= 2:
        separacion = float(np.median(np.diff(filas_ocupadas))) * paso_y
    else:
        separacion = 1.4 * mediana_h
    ancho_celda = paso_x * (1.0 - margen)
    alto_celda = separacion * (1.0 - margen)

    celdas: dict[tuple[int, int], dict] = {}
    for b in blobs:
        cx = b[0] + b[2] / 2.0
        cy = b[1] + b[3] / 2.0
        col = int(round((cx - x0) / paso_x))
        fila = int(round((cy - y0) / paso_y))
        if col < 0 or fila not in ajuste["y_indices"]:
            continue
        clave = (fila, col)
        if clave in celdas:
            continue
        # la retícula marca el CENTRO de la celda, no su esquina
        gx = x0 + col * paso_x
        gy = y0 + fila * paso_y
        h_img, w_img = imagen.shape[:2]
        x = int(round(gx - ancho_celda / 2.0))
        y = int(round(gy - alto_celda / 2.0))
        # recortar a la imagen: un slice negativo en numpy daría basura
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        ancho = max(1, min(int(round(ancho_celda)), w_img - x))
        alto = max(1, min(int(round(alto_celda)), h_img - y))
        celdas[clave] = {
            "x": x,
            "y": y,
            "w": ancho,
            "h": alto,
            "blobs": None,
            "fila": fila,
            "rejilla": True,
        }
    orden = sorted(celdas)
    return [celdas[k] for k in orden]




def _reglas_verticales(tinta_t: np.ndarray, minimo: int) -> list[float]:
    """Columnas de las que están separadas los recuadros."""
    perfil = _perfil_runs(tinta_t)
    return _lineas_del_perfil(perfil, minimo)


def _perfil_runs(tinta: np.ndarray) -> np.ndarray:
    """Longitud del trazo más largo de cada fila de la máscara."""
    h = tinta.shape[0]
    perfil = np.zeros(h, dtype=np.int64)
    for y in range(h):
        fila = tinta[y]
        if not fila.any():
            continue
        bordes = np.flatnonzero(np.diff(np.concatenate(([0], fila, [0]))))
        if len(bordes) >= 2:
            perfil[y] = int((bordes[1::2] - bordes[0::2]).max())
    return perfil


def _lineas_del_perfil(perfil: np.ndarray, minimo: int, tol: float = 2.0) -> list[float]:
    filas = [
        y for y in range(1, len(perfil) - 1) if perfil[y] >= minimo
    ]
    grupos: list[list[int]] = []
    for y in filas:
        if grupos and y - grupos[-1][-1] <= tol:
            grupos[-1].append(y)
        else:
            grupos.append([y])
    salida: list[float] = []
    for g in grupos:
        centro = float(np.mean(g))
        if salida and abs(centro - salida[-1]) <= 3.0:
            salida[-1] = (salida[-1] + centro) / 2.0
        else:
            salida.append(centro)
    return salida


def _lattice_de_reglas(
    xs: list[float], tol: float = 2.5, paso_minimo: float = 0.0
) -> tuple[list[float], float]:
    """Encuentra la retícula regular que mejor explica dónde están las reglas.

    Descartar a mano las reglas sueltas es frágil:取决于 del orden, y con
    pocas reglas buenas el paso se va al valor equivocado y se descarta todo. Aquí
    se prueba cada paso posible y se gana el que más reglas coloca sobre la
    retícula, así que una regla de más no puede arrastrar al resto.
    """
    if len(xs) < 3:
        return [], 0.0
    ref = xs[0]
    limite = (xs[-1] - ref) / 2.0
    if limite < max(6.0, paso_minimo):
        return [], 0.0
    mejor_coincidencias = 0
    mejor_paso = 0.0
    # una casilla tiene que caber un número, así que el paso no puede ser
    # más corto que la cifra; sin este mínimo se gana un paso diminuto que
    # «explica» más reglas porque cae entre las de verdad
    for paso in np.arange(max(6.0, paso_minimo), limite, 0.25):
        n = int((xs[-1] - ref) / paso) + 1
        coincidencias = 0
        i = 0
        for j in range(n):
            objetivo = ref + j * paso
            while i < len(xs) and xs[i] < objetivo - tol:
                i += 1
            if i < len(xs) and abs(xs[i] - objetivo) <= tol:
                coincidencias += 1
                i += 1
        # en empate gana el paso mayor: varias retículas pueden colocar las mismas
        # reglas y la grande es la buena
        if coincidencias >= mejor_coincidencias and paso >= max(6.0, paso_minimo):
            mejor_coincidencias = coincidencias
            mejor_paso = float(paso)
    if mejor_paso <= 0.0 or mejor_coincidencias < 3:
        return [], 0.0
    ref = xs[0]
    n = int((xs[-1] - ref) / mejor_paso) + 1
    utiles: list[float] = []
    i = 0
    for j in range(n):
        objetivo = ref + j * mejor_paso
        while i < len(xs) and xs[i] < objetivo - tol:
            i += 1
        if i < len(xs) and abs(xs[i] - objetivo) <= tol:
            utiles.append(float(xs[i]))
            i += 1
    return utiles, mejor_paso


def _cuadros_por_columnas(
    imagen: np.ndarray, cfg: dict
) -> tuple[list[dict], dict]:
    """Una celda por recuadro: la columna sale de las reglas, la fila de las cifras.

    Las reglas verticales son la única señal limpia de una tabla: atraviesan todo
    el alto de la casilla y salen con un paso muy regular. Las horizontales no,
    porque las rompen los trazos de las cifras de arriba y de abajo, así que la
    fila se deduce de dónde están las propias cifras, que comparten banda
    vertical siempre que comparten casilla.

    Recortar por casilla y no por mancha es lo que arregla los números de dos
    cifras: si «24» llega en dos manchas, recortando por casilla se lee 24, y
    recortando por mancha salían dos celdas, un 2 y un 4.
    """
    meta: dict = {}
    mascara = _mascara_tinta(imagen, cfg)
    if mascara.size == 0:
        return [], meta
    tinta = mascara > 0
    h, w = tinta.shape
    xs = _reglas_verticales(np.ascontiguousarray(tinta.T), max(12, int(h * 0.04)))
    if len(xs) < 4:
        meta["columnas"] = 0
        return [], meta
    meta["reglas_x"] = len(xs)
    alturas = sorted(b[3] for b in _componentes(mascara, 0.00002 * mascara.size))
    mediana_h = alturas[len(alturas) // 2] if alturas else 0.0
    meta["altura_digito"] = int(mediana_h)
    utiles, paso = _lattice_de_reglas(xs, paso_minimo=1.25 * mediana_h)
    if len(utiles) < 4 or paso <= 0.0:
        meta["columnas"] = 0
        return [], meta
    meta["paso"] = round(paso, 1)

    # los bordes de la tabla también acotan una columna, aunque no se dibujen
    bordes = [0.0] + utiles + [float(w)]
    columnas = [
        (bordes[i], bordes[i + 1])
        for i in range(len(bordes) - 1)
        if bordes[i + 1] - bordes[i] >= 4
    ]
    meta["columnas"] = len(columnas)
    if len(columnas) < 2 or mediana_h <= 0:
        return [], meta

    # Las filas salen del perfil de tinta, no de agrupar las manchas: dentro de una
    # fila todas las columnas tienen algo escrito, así que contando cuántas hay con
    # tinta en cada franja de píxeles sale un bloque alto por fila y un pico de un
    # píxel por cada separador. Agrupar manchas fallaba porque _blobs_digitos
    # descarta buena parte de las cifras y entonces una fila se partía en dos.
    umbral = max(2, int(len(columnas) * 0.55))
    conteo = np.zeros(h, dtype=np.int64)
    for x0, x1 in columnas:
        conteo += tinta[:, int(x0):int(x1)].any(axis=1).astype(np.int64)
    alto = conteo >= umbral
    bandas: list[tuple[int, int]] = []
    y = 0
    while y < h:
        if alto[y]:
            y2 = y
            while y2 < h and alto[y2]:
                y2 += 1
            bandas.append((y, y2))
            y = y2
        else:
            y += 1
    # los separadores son de un píxel: las filas son las franjas con altura de cifra
    filas = [b for b in bandas if (b[1] - b[0]) >= max(8, 0.5 * mediana_h)]
    if not filas:
        meta["filas"] = 0
        return [], meta
    meta["filas"] = len(filas)

    margen = float(cfg.get("margen_celda", 0.06))
    celdas: list[dict] = []
    for indice_fila, (y0, y1) in enumerate(filas):
        for x0, x1 in columnas:
            ancho = x1 - x0
            alto_celda = y1 - y0
            respiro = min(ancho, alto_celda) * margen
            celdas.append(
                {
                    "x": int(round(x0 + respiro)),
                    "y": int(round(y0 + respiro)),
                    "w": int(round(ancho - 2 * respiro)),
                    "h": int(round(alto_celda - 2 * respiro)),
                    "blobs": None,
                    "fila": indice_fila,
                }
            )
    meta["celdas"] = len(celdas)
    return celdas, meta


def _asignar_filas(
    blobs: list[tuple[int, int, int, int]], cfg: dict, fusionar: bool = True
) -> list[dict]:
    if not blobs:
        return []
    alturas = sorted(b[3] for b in blobs)
    mediana_h = alturas[len(alturas) // 2]
    filas = _filas(blobs, tolerancia=0.55 * mediana_h)
    celdas = []
    for indice_fila, fila in enumerate(filas):
        grupos = [[fila[0]]]
        if fusionar:
            limite = float(cfg.get("factor_fusion", 0.30)) * mediana_h
            for previo, actual in zip(fila, fila[1:]):
                hueco = actual[0] - (previo[0] + previo[2])
                if hueco <= limite and len(grupos[-1]) < 2:
                    grupos[-1].append(actual)
                else:
                    grupos.append([actual])
        else:
            grupos = [[b] for b in fila]
        for grupo in grupos:
            if len(grupo) == 1:
                x, y, w, h = grupo[0]
                celdas.append(
                    {"x": x, "y": y, "w": w, "h": h, "blobs": None, "fila": indice_fila}
                )
            else:
                caja = _caja_de(grupo)
                if caja is not None:
                    caja["fila"] = indice_fila
                    celdas.append(caja)
    return celdas


def _celdas_por_rejilla(imagen: np.ndarray, cfg: dict) -> list[dict]:
    rejilla = cfg.get("rejilla") or {}
    cols = int(rejilla.get("columnas") or 0)
    if cols <= 0:
        return []
    x0 = int(rejilla.get("x", 0))
    y0 = int(rejilla.get("y", 0))
    paso_x = float(rejilla.get("paso_x") or 0)
    paso_y = float(rejilla.get("paso_y") or 0)
    filas = int(rejilla.get("filas") or 0)
    if paso_x <= 0 or paso_y <= 0:
        return []
    celdas = []
    h_img, w_img = imagen.shape[:2]
    for f in range(filas):
        for c in range(cols):
            x = int(round(x0 + c * paso_x))
            y = int(round(y0 + f * paso_y))
            ancho = int(round(rejilla.get("ancho") or paso_x * 0.92))
            alto = int(round(rejilla.get("alto") or paso_y * 0.92))
            if x < 0 or y < 0 or x + ancho > w_img or y + alto > h_img:
                continue
            celdas.append(
                {"x": x, "y": y, "w": ancho, "h": alto, "blobs": None, "fila": f}
            )
    return celdas


def _umbral_valle(g: np.ndarray) -> float | None:
    ordenados = np.sort(g)
    if ordenados.size < 4:
        return None
    saltos = np.diff(ordenados)
    if saltos.size == 0:
        return None
    j = int(np.argmax(saltos))
    if j + 1 < 2 or (ordenados.size - (j + 1)) < 2:
        return None
    mediana = float(np.median(ordenados))
    if float(saltos[j]) < 0.4 * mediana:
        return None
    return float((ordenados[j] + ordenados[j + 1]) / 2.0)


def _modo_agrupacion(blobs: list[tuple[int, int, int, int]], cfg: dict) -> str:
    if len(blobs) < 4:
        return "celda"
    alturas = sorted(b[3] for b in blobs)
    tolerancia = 0.55 * alturas[len(alturas) // 2]
    huecos: list[float] = []
    for fila in _filas(blobs, tolerancia):
        for a, b in zip(fila, fila[1:]):
            huecos.append(float(b[0] - (a[0] + a[2])))
    if len(huecos) < 4:
        return "celda"
    g = np.array(huecos, dtype=np.float32)
    if g.max() - g.min() < 1.0:
        return "celda"
    return "hueco" if _umbral_valle(g) is not None else "celda"


def _celdas_por_texto(blobs: list[tuple[int, int, int, int]], cfg: dict) -> list[dict]:
    if not blobs:
        return []
    alturas = sorted(b[3] for b in blobs)
    mediana_h = alturas[len(alturas) // 2]
    filas = _filas(blobs, tolerancia=0.55 * mediana_h)
    celdas = []
    for fila in filas:
        for grupo in _grupos_por_hueco(fila, cfg):
            caja = _caja_de(grupo)
            if caja is not None:
                celdas.append(caja)
    return celdas

def _caja_de(blobs: list[tuple[int, int, int, int]]) -> dict | None:
    if not blobs:
        return None
    x0 = min(b[0] for b in blobs)
    y0 = min(b[1] for b in blobs)
    x1 = max(b[0] + b[2] for b in blobs)
    y1 = max(b[1] + b[3] for b in blobs)
    relativos = [(b[0] - x0, b[1] - y0, b[2], b[3]) for b in blobs]
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "blobs": relativos}


def clasificar_color(bgr: np.ndarray, cfg: dict) -> str:
    if bgr.size == 0:
        return "negro"
    hsv = cv2.cvtColor(bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV)
    h = float(np.median(hsv[:, 0, 0])) * 2.0
    s = float(np.median(hsv[:, 0, 1]))
    v = float(np.median(hsv[:, 0, 2]))
    c = cfg["color"]
    if s >= c["s_min"] and v >= c["v_min"]:
        for lo, hi in c["rojo_h"]:
            if lo <= h <= hi:
                return "rojo"
        for lo, hi in c["verde_h"]:
            if lo <= h <= hi:
                return "verde"
    if v <= c["negro_v_max"] and s <= c["negro_s_max"]:
        return "negro"
    return "negro"


def _color_superficie(recorte: np.ndarray, mascaras: list[np.ndarray], cfg: dict) -> str:
    if recorte.size == 0:
        return "negro"
    tinta = np.zeros(recorte.shape[:2], dtype=np.uint8)
    for mascara in mascaras:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tinta = cv2.bitwise_or(tinta, cv2.dilate(mascara, k, iterations=1))
    superficie = recorte[tinta == 0]
    if superficie.shape[0] < 8:
        superficie = recorte.reshape(-1, 3)
    return clasificar_color(superficie, cfg)


def _preparar_digito(mascara: np.ndarray) -> np.ndarray:
    m = cv2.resize(
        mascara, (P.TAMANO_PLANTILLA, P.TAMANO_PLANTILLA), interpolation=cv2.INTER_AREA
    )
    m = m.astype(np.float32) / 255.0
    m = cv2.GaussianBlur(m, (3, 3), 0)
    return (m * 255.0).astype(np.uint8)


def ranking_digitos(mascara: np.ndarray, top: int = 4) -> list[tuple[str, float]]:
    """Dígitos candidatos de una máscara, ordenados de mejor a peor parecido."""
    vectores, etiquetas, _ = P.construir_plantillas()
    vec = P.vector_caracteristicas(_preparar_digito(mascara))
    normas = np.linalg.norm(vectores, axis=1)
    denominador = normas * (float(np.linalg.norm(vec)) or 1.0)
    sim = (vectores @ vec) / np.where(denominador == 0, 1.0, denominador)
    orden = np.argsort(sim)[::-1][:top]
    return [(etiquetas[i], float(sim[i])) for i in orden]


def reconocer_digito(mascara: np.ndarray) -> tuple[str, float]:
    return ranking_digitos(mascara, top=1)[0]


def _separar_doble_digito(mascara: np.ndarray) -> list[tuple[int, np.ndarray]]:
    h, w = mascara.shape[:2]
    if h <= 0 or w <= 0 or w / float(h) < 1.25:
        return [(0, mascara)]
    columnas = mascara.sum(axis=0).astype(np.float32)
    ancho_esperado = float(h)
    centro = w / 2.0
    margen = max(1, int(round(0.30 * ancho_esperado)))
    ini = int(max(0.0, min(centro - ancho_esperado / 2.0 + margen, w - 2)))
    fin = int(max(ini + 2, min(centro + ancho_esperado / 2.0 - margen, w)))
    if fin <= ini + 1:
        ini, fin = int(w * 0.30), int(w * 0.70)
    if fin <= ini + 1:
        return [(0, mascara)]
    valle = int(np.argmin(columnas[ini:fin])) + ini
    izquierda, derecha = mascara[:, :valle], mascara[:, valle:]
    total = float(mascara.sum()) or 1.0
    if izquierda.sum() < 0.18 * total or derecha.sum() < 0.18 * total:
        return [(0, mascara)]
    if columnas[valle] > 0.22 * float(columnas.max() or 1.0):
        return [(0, mascara)]
    return [(0, izquierda), (valle, derecha)]


def _mascaras_celda(recorte: np.ndarray, blobs) -> list[tuple[int, int, np.ndarray]]:
    if blobs:
        salida = []
        for bx, by, bw, bh in blobs:
            sub = recorte[by : by + bh, bx : bx + bw]
            if sub.size:
                salida.append((bx, by, _mascara_digitos_oscuros(sub)))
        return salida

    digitos = _mascara_digitos_oscuros(recorte)
    area_min = max(4.0, 0.0004 * recorte.shape[0] * recorte.shape[1])
    num, etiquetas, stats, _ = cv2.connectedComponentsWithStats(digitos, 8)
    candidatas = []
    for i in range(1, num):
        bx, by, bw, bh, area = stats[i]
        if area < area_min or bw < 3 or bh < 6 or bw / float(bh) > 3.6:
            continue
        candidatas.append(
            (bx, by, (etiquetas[by : by + bh, bx : bx + bw] == i).astype(np.uint8) * 255)
        )
    if len(candidatas) > 2:
        alturas = sorted(m.shape[0] for _, _, m in candidatas)
        mediana = alturas[len(alturas) // 2]
        candidatas = [
            (x, y, m) for x, y, m in candidatas if 0.42 * mediana <= m.shape[0] <= 2.4 * mediana
        ]
    desdobradas: list[tuple[int, int, np.ndarray]] = []
    for x, y, mascara in candidatas:
        for dx, parte in _separar_doble_digito(mascara):
            if parte.shape[0] >= 6 and parte.shape[1] >= 3:
                desdobradas.append((x + dx, y, parte))
    return desdobradas


def _color_superficie(recorte: np.ndarray, mascaras, cfg: dict) -> str:
    if recorte.size == 0:
        return "negro"
    tinta = np.zeros(recorte.shape[:2], dtype=np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for x, y, mascara in mascaras:
        dilatada = cv2.dilate(mascara, k, iterations=1)
        h_dest, w_dest = tinta.shape
        alto, ancho = dilatada.shape
        if y >= h_dest or x >= w_dest:
            continue
        y1, x1 = min(h_dest, y + alto), min(w_dest, x + ancho)
        tinta[y:y1, x:x1] = cv2.bitwise_or(
            tinta[y:y1, x:x1], dilatada[: y1 - y, : x1 - x]
        )
    superficie = recorte[tinta == 0]
    if superficie.shape[0] < 8:
        superficie = recorte.reshape(-1, 3)
    return clasificar_color(superficie, cfg)


def _reconocer_celda(imagen: np.ndarray, caja: dict, cfg: dict) -> Celda | None:
    x, y, w, h = caja["x"], caja["y"], caja["w"], caja["h"]
    recorte = imagen[y : y + h, x : x + w]
    if recorte.size == 0:
        return None
    mascaras = _mascaras_celda(recorte, caja.get("blobs"))
    if not mascaras:
        return None
    color = _color_superficie(recorte, mascaras, cfg)

    mascaras.sort(key=lambda par: par[0])
    recognitions = [reconocer_digito(m) for _, _, m in mascaras]
    texto = "".join(r[0] for r in recognitions)
    confianza = sum(r[1] for r in recognitions) / len(recognitions)
    numero = int(texto) if texto.isdigit() and len(texto) <= 2 else None
    if numero is not None and not (0 <= numero <= 36):
        numero = None

    reparado = False
    if numero is None and len(mascaras) == 2:
        # "38" o "39" no existen en la ruleta: el primer glifo se suele leer
        # como 3 cuando es un 1 con banderola. Se reasigna cada glifo a su
        # mejor candidato y se acepta solo el resultado que cae en 10..36.
        opciones = [ranking_digitos(m, top=5) for _, _, m in mascaras]
        mejor: tuple[float, str] | None = None
        for izq, score_izq in opciones[0]:
            for der, score_der in opciones[1]:
                candidato = int(izq + der)
                if not 10 <= candidato <= 36:
                    continue
                total = score_izq + score_der
                if mejor is None or total > mejor[0]:
                    mejor = (total, izq + der)
        if mejor is not None and mejor[1] != texto:
            texto = mejor[1]
            numero = int(texto)
            confianza = round(mejor[0] / 2.0, 4)
            reparado = True

    coincide = numero is not None and R.color_de(numero) == color
    if numero is not None and not coincide:
        color = R.color_de(numero)
    return Celda(
        numero=numero,
        color=color,
        confianza=round(confianza, 4),
        digitos=texto,
        x=x,
        y=y,
        w=w,
        h=h,
        fila=caja.get("fila", 0),
        color_coincide=coincide,
        reparado=reparado,
    )


# Cada pase usa una máscara de tinta y un recorte de celda distintos. La
# retícula se calcula una sola vez y se comparte, así todos los pasos leen
# exactamente las mismas celdas y se pueden comparar número a número.
#
#   base           configuración normal
#   margen-amplio   celda más ancha y umbral bajo: recupera dígitos tenues
#                  que se quedaban fuera, pero arriesga a coger al vecino
#   margen-ajustado recorte más ceñido y umbral alto: descarta restos del
#                  vecino y suelta los dígitos aislados por el ruido
VARIANTES_MULTIPASE = (
    {"nombre": "base", "margen": 0.06, "umbral": 1.00},
    {"nombre": "margen-amplio", "margen": 0.00, "umbral": 0.82},
    {"nombre": "margen-ajustado", "margen": 0.13, "umbral": 1.22},
)



def _celdas_de_un_paso(
    imagen: np.ndarray,
    cfg: dict,
    ajuste: dict,
    margen: float,
    factor_umbral: float,
) -> dict[tuple[int, int], Celda]:
    """Un pase: máscara de tinta + recorte con este margen → celdas por posición."""
    cfg_paso = dict(cfg)
    cfg_paso["margen_celda"] = margen
    cfg_paso["umbral_tinta"] = max(
        float(cfg.get("umbral_minimo", 6.0)),
        float(cfg["umbral_tinta"]) * factor_umbral,
    )
    blobs = _blobs_digitos(_mascara_tinta(imagen, cfg_paso))
    cajas = _celdas_por_rejilla_inferida(imagen, blobs, cfg_paso, ajuste)
    salida: dict[tuple[int, int], Celda] = {}
    for caja in cajas:
        celda = _reconocer_celda(imagen, caja, cfg_paso)
        if celda is None or not celda.digitos:
            continue
        col = int(round((caja["x"] + caja["w"] / 2.0 - ajuste["x"]) / ajuste["paso_x"]))
        salida[(celda.fila, col)] = celda
    return salida


def _votar_celda(
    clave: tuple[int, int],
    celdas_paso: list[dict[tuple[int, int], Celda]],
    total_pases: int,
) -> Celda | None:
    """Combina lo que vio cada pase en la misma celda."""
    candidatas = [p[clave] for p in celdas_paso if clave in p]
    if not candidatas:
        return None
    conteo: dict[int | None, int] = {}
    for c in candidatas:
        conteo[c.numero] = conteo.get(c.numero, 0) + 1
    if len(conteo) == 1:
        ganador = candidatas[0]
        celda = Celda(
            numero=ganador.numero, color=ganador.color, confianza=ganador.confianza,
            digitos=ganador.digitos, x=ganador.x, y=ganador.y, w=ganador.w, h=ganador.h,
            fila=ganador.fila, color_coincide=ganador.color_coincide,
            alternativa=ganador.alternativa, reparado=ganador.reparado,
            votos=len(candidatas), pases_totales=total_pases, desacuerdos=[],
        )
        if len(candidatas) < total_pases:
            celda.confianza = round(celda.confianza * 0.72, 4)
        return celda
    mejor = max(conteo.items(), key=lambda kv: (kv[1], kv[0] is not None))
    numero = mejor[0]
    representante = next((c for c in candidatas if c.numero == numero), candidatas[0])
    otros = sorted({c.numero for c in candidatas if c.numero != numero and c.numero is not None})
    celda = Celda(
        numero=numero, color=representante.color, confianza=representante.confianza,
        digitos=representante.digitos, x=representante.x, y=representante.y,
        w=representante.w, h=representante.h, fila=representante.fila,
        color_coincide=representante.color_coincide,
        alternativa=otros[0] if otros else None, reparado=representante.reparado,
        votos=mejor[1], pases_totales=total_pases, desacuerdos=otros,
    )
    celda.confianza = round(celda.confianza * (0.6 + 0.2 * mejor[1]), 4)
    return celda




def analizar_imagen_multipase(
    imagen: np.ndarray,
    cfg: dict,
    ajuste: dict,
    max_pasos: int = 3,
) -> tuple[list[Celda], dict]:
    """Analiza la misma imagen hasta `max_pasos` veces y contrasta los resultados."""
    variantes = VARIANTES_MULTIPASE[: max(1, min(3, max_pasos))]
    pasos = [
        _celdas_de_un_paso(imagen, cfg, ajuste, v["margen"], v["umbral"])
        for v in variantes
    ]
    total = len(pasos)
    claves: set[tuple[int, int]] = set()
    for p in pasos:
        claves |= set(p)
    celdas: list[Celda] = []
    for clave in sorted(claves):
        c = _votar_celda(clave, pasos, total)
        if c is not None and c.digitos:
            celdas.append(c)
    coinciden = sum(1 for c in celdas if c.votos == total and not c.desacuerdos)
    celdas_solo_uno = sum(1 for c in celdas if c.votos == 1 and total > 1)
    desacuerdan = sum(1 for c in celdas if c.desacuerdos)
    meta = {
        "pasos": total,
        "variantes": [v["nombre"] for v in variantes],
        "celdas": len(celdas),
        "coinciden": coinciden,
        "celdas_solo_uno": celdas_solo_uno,
        "desacuerdan": desacuerdan,
        "acuerdo_pct": round(100.0 * coinciden / len(celdas), 1) if celdas else 0.0,
    }
    return celdas, meta


def analizar_imagen(ruta: Path, cfg: dict | None = None) -> ResultadoImagen:
    cfg = cfg or cargar_config()
    resultado = ResultadoImagen(archivo=ruta.name, ruta=str(ruta), modo="desconocido")
    imagen = cargar_imagen(ruta)
    if imagen is None:
        resultado.errores.append("No se pudo decodificar la imagen")
        return resultado
    escala = float(cfg.get("escala") or 1.0)
    if abs(escala - 1.0) > 1e-6:
        imagen = cv2.resize(imagen, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    imagen = aplicar_roi(imagen, cfg.get("roi"))
    if imagen.size == 0:
        resultado.errores.append("El ROI configurado deja la imagen vacía")
        return resultado

    ajuste_global: dict | None = None
    modo = cfg.get("modo", "auto")
    if modo not in {"auto", "celda", "hueco", "rejilla"}:
        modo = "auto"

    if modo == "rejilla":
        intentos = [("rejilla", _celdas_por_rejilla(imagen, cfg))]
    elif modo == "hueco":
        blobs = _blobs_digitos(_mascara_tinta(imagen, cfg))
        intentos = [("hueco", _celdas_por_hueco(blobs, cfg))]
    elif modo == "celda":
        blobs = _blobs_digitos(_mascara_tinta(imagen, cfg))
        intentos = [("celda", _asignar_filas(blobs, cfg, fusionar=False))]
    else:
        blobs = _blobs_digitos(_mascara_tinta(imagen, cfg))
        ajuste_global = _ajustar_rejilla_2d(blobs, cfg)
        if ajuste_global is not None:
            # se guarda la retícula propuesta: la UI la usa como base para
            # sugerir una calibración manual si el residuo sale alto
            resultado.rejilla = {
                "x": float(ajuste_global["x"]),
                "y": float(ajuste_global["y"]),
                "paso_x": float(ajuste_global["paso_x"]),
                "paso_y": float(ajuste_global["paso_y"]),
                "columnas": int(ajuste_global["columnas"]),
                "filas": int(ajuste_global["filas"]),
                "residuo_rel": float(ajuste_global["residuo_rel"]),
                "confianza": float(ajuste_global["confianza"]),
            }
        celdas_cuadros, meta_cuadros = _cuadros_por_columnas(imagen, cfg)
        resultado.tabla = meta_cuadros
        intentos = [
            ("rejilla", _celdas_por_rejilla_inferida(imagen, blobs, cfg, ajuste_global)),
            ("hueco", _celdas_por_hueco(blobs, cfg)),
            ("celda", _asignar_filas(blobs, cfg, fusionar=False)),
        ]

    mejor: tuple[str, list[Celda]] | None = None
    respaldo: tuple[str, list[Celda]] | None = None
    meta_pasos: dict | None = None

    max_pasos = int(cfg.get("max_pasos", 3) or 1)
    # Los recuadros son la rejilla cuando los números van dentro de ellos: las
    # reglas dan las columnas y el perfil de tinta las filas, así que cada celda se
    # recorta exactamente y no hace falta adivinar un paso. Pero no se usa siempre:
    # un historial de fichas también tiene columnas y ahí esta lectura no vale.
    # Se queda solo si deja muchas más cifras que el resto, que es lo que pasa en
    # las capturas nuevas (86 leídas por 196 casillas).
    celdas_cuadros_leidas: list[Celda] = []
    if (
        modo in {"auto", "rejilla"}
        and meta_cuadros.get("celdas", 0) >= 6
        and meta_cuadros.get("filas", 0) >= 4
    ):
        celdas_cuadros_leidas = [
            celda
            for celda in (
                _reconocer_celda(imagen, caja, cfg) for caja in celdas_cuadros
            )
            if celda is not None and celda.digitos
        ]

    if mejor is None and max_pasos > 1 and ajuste_global is not None and modo in {"auto", "rejilla"}:
        celdas, meta_pasos = analizar_imagen_multipase(
            imagen, cfg, ajuste_global, max_pasos
        )
        if celdas:
            mejor = ("multipase", celdas)
            resultado.multipase = meta_pasos
    if mejor is None:
        for nombre, cajas in intentos:
            celdas = []
            for caja in cajas:
                celda = _reconocer_celda(imagen, caja, cfg)
                if celda is not None and celda.digitos:
                    celdas.append(celda)
            if not celdas:
                continue
            validas = [c for c in celdas if c.numero is not None]
            if respaldo is None and validas:
                respaldo = (nombre, celdas)
            if len(validas) < 2:
                continue
            media = sum(c.confianza for c in validas) / len(validas)
            if media >= 0.35:
                mejor = (nombre, celdas)
                break

    if mejor is None:
        mejor = respaldo

    # el método de los recuadros solo gana si deja claramente más cifras
    if celdas_cuadros_leidas:
        validas_cuadros = [c for c in celdas_cuadros_leidas if c.numero is not None]
        media_cuadros = (
            sum(c.confianza for c in validas_cuadros) / len(validas_cuadros)
            if validas_cuadros
            else 0.0
        )
        if validas_cuadros and media_cuadros >= 0.35:
            # Se comparan las cifras que se quedan, no las detectadas: por debajo
            # del umbral se descartan igual. Contando las detectadas, el multipase
            # «ganaba» la segunda captura con 189 leídas de las que solo pasaban 61,
            # y el resultado final eran 377 en vez de 495.
            umbral = float(cfg["umbral_confianza"])
            quedan_cuadros = [
                c for c in validas_cuadros if c.confianza >= umbral
            ]
            _, celdas_actuales = mejor if mejor is not None else ("", [])
            quedan_actuales = [
                c
                for c in celdas_actuales
                if c.numero is not None and c.confianza >= umbral
            ]
            if len(quedan_cuadros) >= 1.25 * max(len(quedan_actuales), 1):
                mejor = ("cuadros", celdas_cuadros_leidas)
        if mejor is not None and mejor[0] == "cuadros" and max_pasos > 1:
            resultado.multipase = {
                "pasos": max_pasos,
                "variantes": ["cuadros"],
                "celdas": len(celdas_cuadros_leidas),
                "coinciden": len(validas_cuadros),
                "celdas_solo_uno": 0,
                "desacuerdan": 0,
                "acuerdo_pct": 100.0,
            }
    if mejor is None:
        resultado.advertencia = (
            "No se detectaron números. Ajusta config/deteccion.json "
            "(umbral_tinta, escala o una rejilla manual)."
        )
        return resultado

    nombre_modo, celdas = mejor
    resultado.modo = nombre_modo
    celdas.sort(key=lambda c: (c.fila, c.x))
    if cfg.get("invertir_orden"):
        celdas.reverse()

    umbral = float(cfg["umbral_confianza"])
    n_desacuerdo = 0
    n_flojo = 0
    n_reparado = 0
    for c in celdas:
        resultado.celdas.append(c.a_dict())
        if c.numero is not None:
            resultado.tiradas.append(c.numero)
            if c.confianza < umbral:
                resultado.n_dudas += 1
            if c.desacuerdos:
                n_desacuerdo += 1
            elif c.pases_totales > 1 and c.votos == 1:
                n_flojo += 1
            if c.reparado:
                n_reparado += 1
            if not c.color_coincide:
                resultado.n_conflicto_color += 1
    resultado.n_detectadas = len(celdas)
    resultado.n_reconocidas = len(resultado.tiradas)
    if celdas:
        resultado.confianza_promedio = round(
            sum(c.confianza for c in celdas) / len(celdas), 4
        )
    if resultado.n_dudas:
        resultado.advertencia = (
            f"{resultado.n_dudas} celda(s) bajo el umbral de {umbral:.0%}. Revísalas o ajusta la config."
        )
    if meta_pasos and meta_pasos.get("pasos", 1) > 1:
        resultado.advertencia = (
            (resultado.advertencia + " ") if resultado.advertencia else ""
        ) + (
            f"Doble chequeo de {meta_pasos['pasos']} pasadas: {meta_pasos['coinciden']} de "
            f"{meta_pasos['celdas']} celdas coinciden ({meta_pasos['acuerdo_pct']}%). "
            f"{n_desacuerdo} con dígito en disputa, {n_flojo} vista(s) por un solo pase."
        )
    if n_reparado:
        resultado.advertencia = (
            (resultado.advertencia + " ") if resultado.advertencia else ""
        ) + (
            f"{n_reparado} número(s) no existenían en la ruleta y se corrigieron "
            f"con el rango válido 0-36 (revísalos: era un dígito ambiguo)."
        )
    if resultado.n_conflicto_color:
        resultado.advertencia = (
            (resultado.advertencia + " ") if resultado.advertencia else ""
        ) + f"{resultado.n_conflicto_color} conflicto(s) color/número (se priorizó el dígito)."

    unidades = [c for c in celdas if len(c.digitos) == 1]
    if resultado.modo in {"celda", "hueco"} and resultado.tiradas and len(unidades) / len(celdas) > 0.9:
        resultado.advertencia = (
            (resultado.advertencia + " ") if resultado.advertencia else ""
        ) + (
            "Casi todas las celdas salieron de 1 dígito: es probable que los dígitos de dos cifras "
            "se estén detectando separados. Prueba con \"modo\": \"hueco\" en config/deteccion.json."
        )
    # Si se leyó menos de tres cuartas partes de lo detectado, la captura no es
    # un historial en retícula y conviene decirlo en vez de estampar una tabla
    # que parece buena y no lo es.
    if celdas and resultado.n_reconocidas / len(celdas) < 0.75:
        resultado.advertencia = (
            (resultado.advertencia + " ") if resultado.advertencia else ""
        ) + (
            f"Solo se leyó {resultado.n_reconocidas} de {len(celdas)} celdas: la captura "
            "no parece un historial en retícula uniforme. Recorta el tablero o revisa "
            "la zona analizada antes de fiarte de estas estadísticas."
        )
    return resultado


EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def listar_imagenes(carpeta: Path) -> list[Path]:
    if not carpeta.is_dir():
        return []
    return sorted(
        p
        for p in carpeta.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONES and not p.name.startswith(".")
    )


def metricas_detector() -> dict:
    return {
        "plantillas": P.info_plantillas(),
        "modos": ["celdas", "texto", "auto"],
        "umbral_confianza": cargar_config()["umbral_confianza"],
    }
