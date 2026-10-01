"""Datos mínimos de la ruleta que necesita el análisis.

Ruleta europea de un solo cero: 37 casillas (0-36), 18 rojas y 18 negras.
El panel de reglas de la ruleta está oculto en la interfaz, así que aquí solo
se conserva lo que el análisis necesita de verdad: el color de cada número y
el reparto en docenas.
"""

from __future__ import annotations

CASILLAS = 37
VENTAJA_CASA = 2.70
RTP = 97.30

NUMEROS_ROJOS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
NUMEROS_NEGROS = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}

DOCENAS = {
    "1-12": list(range(1, 13)),
    "13-24": list(range(13, 25)),
    "25-36": list(range(25, 37)),
}


def _a_int(numero: int | str) -> int | None:
    if numero is None:
        return None
    try:
        return int(numero)
    except (TypeError, ValueError):
        return None


def color_de(numero: int | str) -> str:
    if str(numero) in {"0", "00"}:
        return "verde"
    n = _a_int(numero)
    if n is None:
        return "fuera_de_rango"
    if n in NUMEROS_ROJOS:
        return "rojo"
    if n in NUMEROS_NEGROS:
        return "negro"
    return "fuera_de_rango"


def dozen_de(numero: int | str) -> str | None:
    """Docena a la que pertenece el número. El 0 no pertenece a ninguna."""
    n = _a_int(numero)
    if n is None or n == 0:
        return None
    if n <= 12:
        return "1-12"
    if n <= 24:
        return "13-24"
    return "25-36"


def resumen() -> dict:
    return {
        "casillas": CASILLAS,
        "rojos": len(NUMEROS_ROJOS),
        "negros": len(NUMEROS_NEGROS),
        "verdes": 1,
        "docenas": len(DOCENAS),
        "ventaja_casa": VENTAJA_CASA,
        "rtp": RTP,
    }
