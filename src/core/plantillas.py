from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TAMANO_PLANTILLA = 40

DIRS_FUENTES = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/truetype/liberation2",
    "/usr/share/fonts/truetype/freefont",
]

FUENTES = [
    "DejaVuSans-Bold.ttf",
    "DejaVuSans.ttf",
    "DejaVuSansMono-Bold.ttf",
    "LiberationSans-Bold.ttf",
    "LiberationSans-Regular.ttf",
    "LiberationMono-Bold.ttf",
    "FreeSansBold.ttf",
]

DIGITOS = "0123456789"


@lru_cache(maxsize=1)
def rutas_fuentes() -> list[Path]:
    encontradas = []
    for d in DIRS_FUENTES:
        base = Path(d)
        if not base.is_dir():
            continue
        for nombre in FUENTES:
            p = base / nombre
            if p.is_file():
                encontradas.append(p)
    return encontradas


@lru_cache(maxsize=256)
def _fuente(ruta: str, tamano: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(ruta, tamano)


def _recortar_al_glifo(img: np.ndarray, margen: int = 2) -> np.ndarray:
    binaria = (img > 110).astype(np.uint8)
    coordenadas = cv2.findNonZero(binaria)
    if coordenadas is None or coordenadas.size == 0:
        return img
    x, y, w, h = cv2.boundingRect(coordenadas)
    y0 = max(0, y - margen)
    x0 = max(0, x - margen)
    y1 = min(img.shape[0], y + h + margen)
    x1 = min(img.shape[1], x + w + margen)
    if y1 <= y0 or x1 <= x0:
        return img
    return img[y0:y1, x0:x1]


def _render_digito(digito: str, fuente: Path, tamano_fuente: int = 34) -> np.ndarray:
    render = TAMANO_PLANTILLA * 3
    img = Image.new("L", (render, render), 0)
    draw = ImageDraw.Draw(img)
    font = _fuente(str(fuente), tamano_fuente)
    caja = draw.textbbox((0, 0), digito, font=font)
    ancho = max(1, caja[2] - caja[0])
    alto = max(1, caja[3] - caja[1])
    x = (render - ancho) // 2 - caja[0]
    y = (render - alto) // 2 - caja[1]
    draw.text((x, y), digito, font=font, fill=255)
    recortado = _recortar_al_glifo(np.array(img))
    return cv2.resize(recortado, (TAMANO_PLANTILLA, TAMANO_PLANTILLA), interpolation=cv2.INTER_AREA)


def _normalizar(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    if m.max() - m.min() < 1e-6:
        return np.zeros_like(m)
    m = (m - m.min()) / (m.max() - m.min())
    return m


def _momentos_hu(m: np.ndarray) -> np.ndarray:
    binaria = (m > 0.35).astype(np.uint8) * 255
    hu = cv2.HuMoments(cv2.moments(binaria)).ravel()
    return np.sign(hu) * np.log1p(np.abs(hu))


def _perfil(m: np.ndarray) -> np.ndarray:
    filas = m.mean(axis=1)
    cols = m.mean(axis=0)
    p = np.concatenate([filas, cols])
    return _normalizar(p)


def vector_caracteristicas(m: np.ndarray) -> np.ndarray:
    return np.concatenate([_normalizar(m).ravel(), _perfil(m), _momentos_hu(m)])


@lru_cache(maxsize=1)
def construir_plantillas() -> tuple[np.ndarray, list[str], list[str]]:
    fuentes = rutas_fuentes()
    if not fuentes:
        raise RuntimeError(
            "No se encontraron fuentes TrueType en el sistema para generar las plantillas de dígitos."
        )
    imagenes: list[np.ndarray] = []
    etiquetas: list[str] = []
    origen: list[str] = []
    for fuente in fuentes:
        for d in DIGITOS:
            imagenes.append(_render_digito(d, fuente))
            etiquetas.append(d)
            origen.append(fuente.name)
    vectores = np.stack([vector_caracteristicas(im) for im in imagenes])
    return vectores, etiquetas, origen


def info_plantillas() -> dict:
    vectores, etiquetas, origen = construir_plantillas()
    return {
        "plantillas": len(etiquetas),
        "digitos": len(set(etiquetas)),
        "fuentes": sorted(set(origen)),
        "dim_vector": int(vectores.shape[1]),
        "tamano": TAMANO_PLANTILLA,
    }
