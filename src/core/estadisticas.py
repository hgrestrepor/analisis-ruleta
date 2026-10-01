from __future__ import annotations

from collections import Counter, OrderedDict

from core import ruleta_reglas as R

CAMPOS_REPETICION = ["2", "3", "4+"]

# Solo docenas. Las columnas del tapete (1,4,7...) y las calles quedan fuera
# a propósito: el análisis se centra en repeticiones de docenas.
AGRUPACIONES = OrderedDict(
    [
        ("docenas", {"nombre": "Docenas (1-12 / 13-24 / 25-36)", "sustantivo": "docena", "grupos": R.DOCENAS}),
    ]
)


def etiqueta_grupo(agrupacion: str, clave) -> str:
    info = AGRUPACIONES[agrupacion]
    nums = info["grupos"][clave]
    sustantivo = info["sustantivo"]
    if len(nums) > 3:
        return f"{clave} {sustantivo} ({nums[0]}, {nums[1]}, ... {nums[-1]})"
    return f"{clave} {sustantivo} ({', '.join(map(str, nums))})"


def agrupacion_de_numero(numero: int | str, agrupacion: str = "docenas"):
    grupos = AGRUPACIONES[agrupacion]["grupos"]
    for clave, nums in grupos.items():
        if _a_int(numero) in nums:
            return clave
    return None


def _a_int(numero) -> int | None:
    if numero is None:
        return None
    try:
        return int(numero)
    except (TypeError, ValueError):
        return None


def clasificar_repeticion(longitud_racha: int) -> str | None:
    if longitud_racha == 2:
        return "2"
    if longitud_racha == 3:
        return "3"
    if longitud_racha >= 4:
        return "4+"
    return None


def contar_repeticiones(tiradas, agrupacion: str = "docenas", incluir_cero: bool = False) -> dict:
    grupos = AGRUPACIONES[agrupacion]["grupos"]
    tabla = OrderedDict(
        (
            clave,
            {campo: 0 for campo in CAMPOS_REPETICION}
            | {"Total": 0, "secuencias": 0, "racha_max": 0, "pct": 0.0},
        )
        for clave in grupos
    )

    actual = None
    largo = 0
    desde = 0
    numeros: list = []
    rachas: list[dict] = []

    def cerrar():
        nonlocal actual, largo, desde, numeros
        if actual is not None and largo >= 2:
            fila = tabla[actual]
            fila["secuencias"] += 1
            fila["racha_max"] = max(fila["racha_max"], largo)
            campo = clasificar_repeticion(largo)
            if campo:
                fila[campo] += 1
                fila["Total"] += 1
            rachas.append(
                {
                    "docena": actual,
                    "largo": largo,
                    "campo": campo,
                    "desde": desde,
                    "hasta": desde + largo - 1,
                    "numeros": list(numeros),
                }
            )
        actual, largo, desde, numeros = None, 0, 0, []

    for pos, n in enumerate(tiradas):
        clave = agrupacion_de_numero(n, agrupacion) if (incluir_cero or _a_int(n) != 0) else None
        if clave is None:
            cerrar()
            continue
        if clave == actual:
            largo += 1
            numeros.append(n)
        else:
            cerrar()
            actual, largo, desde, numeros = clave, 1, pos, [n]
    cerrar()

    total_global = sum(f["Total"] for f in tabla.values())
    for f in tabla.values():
        f["pct"] = round(100.0 * f["Total"] / total_global, 2) if total_global else 0.0
    return {"tabla": tabla, "rachas": rachas}


def _vacia():
    return {campo: None for campo in CAMPOS_REPETICION} | {"Total": None, "pct": None}


def alternancia_docenas(tiradas) -> dict:
    """Mide cada cuántas veces se intercalan docenas distintas.

    "Intercalar" es un cambio de docena entre dos tiradas consecutivas. Se
    reportan tres cosas distintas que se confunden entre sí: el porcentaje de
    cambios, cada cuántas tiradas ocurre un cambio, y cuántas veces vuelve a
    aparecer una docena tras ausentarse ese número de tiradas.
    """
    seq = [d for d in (agrupacion_de_numero(t, "docenas") for t in tiradas) if d is not None]
    n = len(seq)
    cambios = 0
    for a, b in zip(seq, seq[1:]):
        if a != b:
            cambios += 1
    pares = max(0, n - 1)

    gaps: dict = {}
    for clave in R.DOCENAS:
        ultimo = None
        distancia: list[int] = []
        for i, d in enumerate(seq):
            if d == clave:
                if ultimo is not None:
                    distancia.append(i - ultimo - 1)
                ultimo = i
        if distancia:
            gaps[clave] = {
                "apariciones": len(distancia) + 1,
                "tiradas_entre_apariciones": round(sum(distancia) / len(distancia), 2),
                "max_tiradas_ausente": max(distancia),
            }

    rachas: list[int] = []
    actual = None
    largo = 0
    for d in seq:
        if d == actual:
            largo += 1
        else:
            if largo:
                rachas.append(largo)
            actual, largo = d, 1
    if largo:
        rachas.append(largo)

    return {
        "tiradas_con_docena": n,
        "pares_adyacentes": pares,
        "cambios": cambios,
        "pct_alternancia": round(100.0 * cambios / pares, 2) if pares else 0.0,
        "tiradas_por_cambio": round(pares / cambios, 2) if cambios else None,
        "repeticiones_iguales": pares - cambios,
        "racha_media": round(sum(rachas) / len(rachas), 2) if rachas else 0.0,
        "racha_max": max(rachas) if rachas else 0,
        "por_docena": gaps,
        "esperado_pct_alternancia": 66.67,
    }


def esquema_tablas(agrupacion_por_defecto: str = "docenas") -> dict:
    return {
        "resumen_global": {
            "titulo": "Resumen global",
            "nota": "Métricas alimentadas por el módulo de visión. Pendiente de datos.",
            "columnas": ["Métrica", "Valor", "% del total", "Esperado"],
            "filas": [
                ["Tiradas analizadas", None, None, "—"],
                ["Imágenes procesadas", None, None, "—"],
                ["Números distintos detectados", None, None, "37"],
                ["Veces que salió el 0", None, None, None],
                ["Veces rojo", None, None, None],
                ["Veces negro", None, None, None],
                ["Racha máxima actual", None, None, "—"],
                ["Racha máxima histórica", None, None, "—"],
            ],
        },
        "frecuencia_numeros": {
            "titulo": "Frecuencia por número (0-36)",
            "nota": "Esperado = tiradas / 37. La desviación compara lo observado contra el azar.",
            "columnas": ["Nº", "Docena", "Color", "Veces", "%", "Esperado", "Desviación"],
            "filas": [
                [n, R.dozen_de(n) or "—", R.color_de(n), None, None, None, None]
                for n in range(0, 37)
            ],
        },
        "frecuencia_docenas": {
            "titulo": "Frecuencia por docena",
            "nota": "Apuesta DOUZAINE: 12 números, paga 2:1.",
            "columnas": ["Docena", "Números", "Veces", "%", "Esperado", "Desviación"],
            "filas": [[k, f"{v[0]}–{v[-1]}", None, None, None, None] for k, v in R.DOCENAS.items()],
        },
        "frecuencia_colores": {
            "titulo": "Frecuencia por color",
            "nota": "18 rojos, 18 negros y 1 verde. Esperado: 48.65% / 48.65% / 2.70%.",
            "columnas": ["Color", "Casillas", "Veces", "%", "Esperado", "Desviación"],
            "filas": [
                ["rojo", 18, None, None, "48.65%", None],
                ["negro", 18, None, None, "48.65%", None],
                ["verde (0)", 1, None, None, "2.70%", None],
            ],
        },
        "repeticion_docenas": {
            "titulo": "Cuántas veces se repitió una docena",
            "nota": (
                "Rachas consecutivas de la misma docena. Solo se contabilizan rachas de 2 o más; "
                "el 0 corta la racha porque no pertenece a ninguna docena."
            ),
            "campos": CAMPOS_REPETICION,
            "columnas": ["Docena"] + CAMPOS_REPETICION + ["Total", "% del total"],
            "filas": [
                [etiqueta_grupo("docenas", k)] + [None] * (len(CAMPOS_REPETICION) + 2)
                for k in R.DOCENAS
            ],
            "totales": ["TOTAL"] + [None] * (len(CAMPOS_REPETICION) + 2),
        },
        "alternancia_docenas": {
            "titulo": "Alternancia de docenas",
            "nota": (
                "Un cambio de alternancia = dos tiradas seguidas de distinta docena. "
                "El 0 no tiene docena, así que queda fuera de los pares. "
                "Esperado al azar: 66.67% de cambios."
            ),
            "columnas": ["Métrica", "Valor", "Esperado"],
            "filas": [
                ["Tiradas con docena", None, "—"],
                ["Pares adyacentes comparados", None, "—"],
                ["Cambios de docena", None, "—"],
                ["% de alternancia", None, "66.67%"],
                ["Tiradas por cada cambio", None, "1.50"],
                ["Repeticiones iguales seguidas", None, "—"],
                ["Racha media (misma docena)", None, "1.50"],
                ["Racha máxima (misma docena)", None, "—"],
            ],
            "columnas_docena": ["Docena", "Apariciones", "Tiradas entre apariciones", "Máx. ausente"],
            "filas_docena": [[k, None, None, None] for k in R.DOCENAS],
        },
    }


def resumen_desde_tiradas(tiradas) -> dict:
    tiradas = [t for t in tiradas if _a_int(t) is not None]
    tiradas = [_a_int(t) for t in tiradas]
    n = len(tiradas)
    conteo = Counter(tiradas)
    colores = Counter(R.color_de(t) for t in tiradas)
    rachas, actual, largo = [], None, 0
    for t in tiradas:
        largo = largo + 1 if t == actual else 1
        actual = t
        rachas.append(largo)
    ultimo = tiradas[-1] if tiradas else None
    racha_actual = 0
    for v in reversed(tiradas):
        if v != ultimo:
            break
        racha_actual += 1
    return {
        "total": n,
        "numeros": dict(sorted(conteo.items())),
        "esperado_numero": n / 37.0,
        "colores": dict(colores),
        "docenas": dict(Counter(R.dozen_de(t) for t in tiradas if R.dozen_de(t))),
        "racha_max": max(rachas) if rachas else 0,
        "racha_actual": racha_actual,
        "racha_actual_numero": tiradas[-1] if tiradas else None,
        "numeros_distintos": len(conteo),
        "ceros": conteo.get(0, 0),
        "repeticiones": {k: contar_repeticiones(tiradas, k)["tabla"] for k in AGRUPACIONES},
        "rachas_detalle": contar_repeticiones(tiradas, "docenas")["rachas"],
        "alternancia_docenas": alternancia_docenas(tiradas),
    }
