from absl import app

import numpy as np
import sympy
from sympy import sin, cos


def main(argv=None):
    # States:
    x, dx, ddx, th, dth, ddth, u = sympy.core.symbol.symbols(
        "x dx ddx th dth ddth u"
    )

    # Constants:
    m_c, m_p, l, g, mu_c, mu_p = sympy.core.symbol.symbols(
        "m_c m_p l g mu_c mu_p"
    )

    sgn = -1.0

    a = (
            (
                (-u - m_p * l * dth ** 2 * (sin(th) + mu_c * sgn * cos(th))) / (m_c + m_p)
            ) + mu_c * g * sgn
    )
    b = (
        l * (4/3 - ((m_p * cos(th)) / (m_c + m_p)) * (cos(th) - mu_c * sgn))
    )

    ddth_expr = (g * sin(th) + cos(th) * a - ((mu_p * dth) / (m_p * l))) / b
    ddth_expr = sympy.simplify(ddth_expr)
    sympy.pretty_print(ddth_expr)

    N_c_expr = (m_c + m_p) * g - m_p * l * (ddth_expr * sin(th) + dth ** 2 * cos(th))

    ddx_expr = (u + m_p * l * (dth ** 2 * sin(th) - ddth_expr * cos(th)) - mu_c * N_c_expr * sgn) / (m_c + m_p)
    ddx_expr = sympy.simplify(ddx_expr)
    sympy.pretty_print(ddx_expr)


if __name__ == "__main__":
    app.run(main)
