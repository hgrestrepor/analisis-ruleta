"""Simulador de la estrategia de repetición de docenas, en pesos colombianos.

Regla simulada
--------------
Cuando dos tiradas consecutivas caen en la misma docena, se apuesta a *todos los
números que no son de esa docena* más el cero. En el ejemplo (5 y luego 11, ambos
en 1-12) son los 24 números de 13-36 más el 0: 25 números en total.

Cálculo de una apuesta (ruleta europea, cero, pago 35:1 en número simple)
    Apostado            = 25 números × 2 500 = 62 500
    Si acierta uno:     gana 35 × 2 500 = 87 500
                        pierde los otros 24 × 2 500 = 60 000
                        neto = +27 500
    Si no acierta:      el 0 y los 12 números de la decidedly repetida salen
                        neto = −62 500

El 0 cuenta como número apostado: si sale, la apuesta gana.

Nota importante: la esperanza matemática es negativa. Acertar un 25/37 paga
87 500 a costa de 62 500 jugados, así que por cada 100 pesos jugados se
recuperan 67,57 en promedio. El simulador muestra el resultado histórico real,
no una ganancia esperada.
"""

from __future__ import annotations

PAGO_NUMERO_SIMPLE = 35  # 35:1 sobre la apuesta
MONEDA = "COP"


def _docena_de(n: int) -> str | None:
    if n == 0:
        return None
    if 1 <= n <= 12:
        return "1-12"
    if 13 <= n <= 24:
        return "13-24"
    if 25 <= n <= 36:
        return "25-36"
    return None


def cop(monto: float) -> str:
    """Formato colombiano: 1.234.567"""
    return f"{round(monto):,}".replace(",", ".")


def numeros_apostados(docena: str, incluir_cero: bool = True) -> list[int]:
    """Números que se betean al detectarse la repetición de `docena`."""
    numeros = [n for n in range(1, 37) if _docena_de(n) != docena]
    if incluir_cero:
        numeros.insert(0, 0)
    return sorted(numeros)


def netos(apuesta: int, cuantos: int) -> tuple[int, int]:
    """(neto si acierta, neto si falla) para una apuesta a `cuantos` números.

    Al acertar, ese número devuelve la apuesta más 35:1, y los otros
    `cuantos - 1` se pierden:
        2 500 × 36 − 62 500 = +27 500
    """
    gana = apuesta * (1 + PAGO_NUMERO_SIMPLE) - apuesta * cuantos
    return gana, -apuesta * cuantos


def _rachas(tiradas: list[int]) -> list[dict]:
    """Rachas de docenas consecutivas, con su posición en la secuencia.

    El 0 no pertenece a ninguna docena, así que parte la racha en dos. Solo se
    devuelven las rachas de 2 o más, que son las que pueden disparar apuesta.
    """
    salida: list[dict] = []
    actual: str | None = None
    largo = 0
    desde = 0
    numeros: list[int] = []

    def cerrar() -> None:
        nonlocal actual, largo, desde, numeros
        if actual is not None and largo >= 2:
            salida.append(
                {
                    "docena": actual,
                    "largo": largo,
                    "desde": desde,
                    "hasta": desde + largo - 1,
                    "numeros": list(numeros),
                }
            )
        actual, largo, desde, numeros = None, 0, 0, []

    for pos, n in enumerate(tiradas):
        d = _docena_de(n)
        if d is None:
            cerrar()
            continue
        if d == actual:
            largo += 1
            numeros.append(n)
        else:
            cerrar()
            actual, largo, desde, numeros = d, 1, pos, [n]
    cerrar()
    return salida


def simular(
    tiradas,
    apuesta: int = 2500,
    incluir_cero: bool = True,
    disparo: int = 2,
    racha_max: int | None = None,
) -> dict:
    """Recorre la secuencia y aplica la regla en cada repetición de docena.

    `disparo` es cuántas repeticiones seguidas de la misma docena hacen falta
    para abrir la apuesta: 2, 3 o 4. `racha_max` descarta las rachas más largas
    que ese valor, porque una racha tan larga ya no es una repetición normal.

    Solo puede haber una apuesta abierta a la vez: se abre tras la repetición y
    se resuelve con la tirada siguiente, sea cual sea.
    """
    tiradas = [int(t) for t in tiradas]
    apuesta = int(apuesta)
    disparo = int(disparo)
    if apuesta <= 0:
        raise ValueError("La apuesta debe ser mayor que cero.")
    if disparo < 2:
        raise ValueError("El disparo necesita al menos 2 repeticiones.")

    # una racha dispara solo en su elemento `disparo`, y nunca si la racha
    # entera es más larga que el tope: esas se descartan por completo
    aperturas: dict[int, dict] = {}
    descartadas: list[dict] = []
    for r in _rachas(tiradas):
        if racha_max is not None and r["largo"] > racha_max:
            descartadas.append(r)
            continue
        pos = r["desde"] + disparo - 1
        if pos <= r["hasta"]:
            aperturas[pos] = r

    movimientos: list[dict] = []
    abierta: dict | None = None
    docena_previa: str | None = None

    for i, n in enumerate(tiradas):
        docena = _docena_de(n)

        # 1) resolver la apuesta pendiente con esta tirada
        if abierta is not None:
            acierto = n in abierta["numeros"]
            total_apostado = apuesta * len(abierta["numeros"])
            gana, pierde = netos(apuesta, len(abierta["numeros"]))
            neto = gana if acierto else pierde
            # lo que la banca devuelve sobre el número ganador: la apuesta más el 35:1
            devuelto = apuesta * (1 + PAGO_NUMERO_SIMPLE) if acierto else 0
            abierta.update(
                {
                    "resolucion_posicion": i,
                    "resultado": n,
                    "acierto": acierto,
                    "apostado": total_apostado,
                    "devuelto": devuelto,
                    "neto": neto,
                }
            )
            movimientos.append(abierta)
            abierta = None

        # 2) abrir apuesta si esta tirada completa la repetición
        if not abierta and i in aperturas:
            r = aperturas[i]
            numeros = numeros_apostados(r["docena"], incluir_cero)
            abierta = {
                "apertura_posicion": i,
                "docena_repetida": r["docena"],
                "numeros": numeros,
                "cuantos_numeros": len(numeros),
                "apuesta_unitaria": apuesta,
                "racha_previa": r["largo"],
                "racha_desde": r["desde"],
            }

        # 3) el 0 no pertenece a ninguna docena: rompe la cadena
        docena_previa = docena

    # una apuesta abierta al final nunca llegó a resolverse
    if abierta is not None:
        abierta.update(
            {
                "resolucion_posicion": None,
                "resultado": None,
                "acierto": None,
                "apostado": apuesta * abierta["cuantos_numeros"],
                "devuelto": 0,
                "neto": 0,
                "sin_resolver": True,
            }
        )
        movimientos.append(abierta)

    resueltas = [m for m in movimientos if not m.get("sin_resolver")]
    ganadoras = [m for m in resueltas if m["acierto"]]
    perdedoras = [m for m in resueltas if not m["acierto"]]

    total_apostado = sum(m["apostado"] for m in resueltas)
    neto = sum(m["neto"] for m in resueltas)
    aciertos = len(ganadoras)

    cuantos = len(numeros_apostados("1-12", incluir_cero))
    prob_teorica = cuantos / 37
    gana, pierde = netos(apuesta, cuantos)
    ev_por_apuesta = prob_teorica * gana + (1 - prob_teorica) * pierde

    return {
        "moneda": MONEDA,
        "apuesta": apuesta,
        "incluye_cero": incluir_cero,
        "disparo": disparo,
        "racha_max": racha_max,
        "rachas_descartadas": len(descartadas),
        "rachas_usadas": len({id(v) for v in aperturas.values()}),
        "numeros_por_apuesta": cuantos,
        "neto_si_gana": gana,
        "neto_si_pierde": pierde,
        "apuestas": len(resueltas),
        "apuestas_sin_resolver": len(movimientos) - len(resueltas),
        "aciertos": aciertos,
        "fallos": len(perdedoras),
        "tasa_acerto": round(100.0 * aciertos / len(resueltas), 2) if resueltas else None,
        "total_apostado": total_apostado,
        "total_devuelto": sum(m["devuelto"] for m in resueltas),
        "neto": neto,
        "roi_pct": round(100.0 * neto / total_apostado, 2) if total_apostado else None,
        "mejor": max((m["neto"] for m in resueltas), default=0),
        "peor": min((m["neto"] for m in resueltas), default=0),
        "por_docena": _resumen_por_docena(resueltas),
        "movimientos": movimientos,
        "regla": _texto_regla(apuesta, incluir_cero, gana, pierde, disparo, racha_max),
        "esperanza_teorica_por_apuesta": round(ev_por_apuesta),
        "prob_teorica_acierto": round(100 * prob_teorica, 2),
        "aviso": (
            f"La esperanza matemática de esta estrategia es negativa: {cop(ev_por_apuesta)} COP "
            "por apuesta en promedio. Es un experimento histórico, no un método rentable."
        ),
    }


def _texto_regla(apuesta, incluir_cero, gana, pierde, disparo, racha_max) -> str:
    if racha_max is not None and racha_max <= disparo:
        return (
            f"El tope de racha ({racha_max}) no supera el disparo ({disparo}): "
            "ninguna racha cumple las dos condiciones, así que no se apuesta."
        )
    otros = (
        "los otros 24 números más el 0 (25 números en total)"
        if incluir_cero
        else "los otros 24 números"
    )
    texto = (
        f"Cuando la misma docena sale {disparo} veces seguidas, se betean {otros} "
        f"a {cop(apuesta)} cada uno. "
    )
    if racha_max is not None:
        texto += f"Las rachas de más de {racha_max} repeticiones se descartan y no se apuesta. "
    return texto + (
        f"Si acierta, neto +{cop(gana)} COP. Si falla, neto {cop(pierde)} COP."
    )


def _resumen_por_docena(movimientos) -> dict:
    salida: dict = {}
    for m in movimientos:
        d = m["docena_repetida"]
        fila = salida.setdefault(d, {"apuestas": 0, "aciertos": 0, "neto": 0})
        fila["apuestas"] += 1
        if m.get("acierto"):
            fila["aciertos"] += 1
        fila["neto"] += m.get("neto", 0)
    return salida
