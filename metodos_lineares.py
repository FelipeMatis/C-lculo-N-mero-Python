# ===============================================================
# Rotinas para resolver sistemas lineares
# Retornos:
#   x, tempo, status
# ou
#   x, tempo, status, passos  (quando retornar_passos=True)
# ===============================================================

import numpy as np
import time

EPS = 1e-18  # tolerância numérica

# ---------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------

def eh_quadrada(matriz):
    """Verifica se a matriz é quadrada."""
    matriz = np.array(matriz)
    return matriz.ndim == 2 and matriz.shape[0] == matriz.shape[1]


def eh_definida_positiva(matriz):
    """Verifica se a matriz é definida positiva via decomposição de Cholesky."""
    matriz = np.array(matriz, dtype=float)
    if not eh_quadrada(matriz):
        return False
    try:
        np.linalg.cholesky(matriz)
        return True
    except np.linalg.LinAlgError:
        return False


def _empacotar_retorno(x, tempo, status, passos=None, retornar_passos=False):
    """Padroniza o formato de retorno."""
    if retornar_passos:
        return x, tempo, status, passos
    else:
        return x, tempo, status

# ---------------------------------------------------------------
# Eliminação de Gauss (sem pivoteamento)
# ---------------------------------------------------------------

def eliminacao_gauss(A, b, retornar_passos=False, mostrar_matrizes=False, **kwargs):
    inicio = time.time()
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    passos = {"matrizes": [], "acoes": []}

    if not eh_quadrada(A) or A.shape[0] != b.shape[0]:
        status = "ERRO: A não é quadrada ou tem dimensões incompatíveis com b."
        return _empacotar_retorno(None, 0.0, status, passos, retornar_passos)

    n = A.shape[0]
    M = np.hstack([A.copy(), b.reshape(-1, 1)])
    if mostrar_matrizes:
        passos["matrizes"].append(("Inicial (A|b)", M.copy()))

    for i in range(n):
        if abs(M[i, i]) < EPS:
            status = f"ERRO: Pivô (linha {i}) muito próximo de zero — pivoteamento necessário."
            return _empacotar_retorno(None, time.time() - inicio, status, passos, retornar_passos)
        for j in range(i + 1, n):
            multiplicador = M[j, i] / M[i, i]
            M[j, i:] -= multiplicador * M[i, i:]
            if mostrar_matrizes:
                passos["acoes"].append(f"Eliminou linha {j} usando linha {i} (m={multiplicador:.6g})")
                passos["matrizes"].append((f"Após eliminação i={i}, j={j}", M.copy()))

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(M[i, i]) < EPS:
            status = f"ERRO: Pivô zero durante retrosubstituição (linha {i})."
            return _empacotar_retorno(None, time.time() - inicio, status, passos, retornar_passos)
        x[i] = (M[i, n] - np.dot(M[i, i + 1:], x[i + 1:])) / M[i, i]

    tempo = time.time() - inicio
    status = "Sucesso (Eliminação de Gauss sem pivoteamento)."
    return _empacotar_retorno(x, tempo, status, passos, retornar_passos)

# ---------------------------------------------------------------
# Pivoteamento parcial (troca de linhas)
# ---------------------------------------------------------------

def pivoteamento_parcial(A, b, retornar_passos=False, mostrar_matrizes=False, **kwargs):
    inicio = time.time()
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    passos = {"matrizes": [], "acoes": []}

    if not eh_quadrada(A) or A.shape[0] != b.shape[0]:
        status = "ERRO: A não é quadrada ou dimensões incompatíveis com b."
        return _empacotar_retorno(None, 0.0, status, passos, retornar_passos)

    n = len(b)
    M = np.hstack([A.copy(), b.reshape(-1, 1)])
    if mostrar_matrizes:
        passos["matrizes"].append(("Inicial (A|b)", M.copy()))

    for i in range(n):
        linha_pivo = np.argmax(np.abs(M[i:, i])) + i
        passos["acoes"].append(f"Pivô escolhido (linha {linha_pivo}) para coluna {i}")
        if abs(M[linha_pivo, i]) < EPS:
            status = f"ERRO: Pivô zero (ou quase) na coluna {i}."
            return _empacotar_retorno(None, time.time() - inicio, status, passos, retornar_passos)
        if linha_pivo != i:
            M[[i, linha_pivo], :] = M[[linha_pivo, i], :]
            passos["acoes"].append(f"Trocou linha {i} com {linha_pivo}")
            if mostrar_matrizes:
                passos["matrizes"].append((f"Após troca {i}<->{linha_pivo}", M.copy()))
        for j in range(i + 1, n):
            multiplicador = M[j, i] / M[i, i]
            M[j, i:] -= multiplicador * M[i, i:]
            passos["acoes"].append(f"Eliminou linha {j} usando linha {i} (m={multiplicador:.6g})")
            if mostrar_matrizes:
                passos["matrizes"].append((f"Após eliminação i={i}, j={j}", M.copy()))

    # Retrosubstituição
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(M[i, i]) < EPS:
            status = f"ERRO: Pivô zero na retrosubstituição (linha {i})."
            return _empacotar_retorno(None, time.time() - inicio, status, passos, retornar_passos)
        x[i] = (M[i, n] - np.dot(M[i, i + 1:], x[i + 1:])) / M[i, i]

    tempo = time.time() - inicio
    status = "Sucesso (Gauss com pivoteamento parcial)."
    return _empacotar_retorno(x, tempo, status, passos, retornar_passos)

# ---------------------------------------------------------------
# Pivoteamento completo (linhas e colunas)
# ---------------------------------------------------------------

def pivoteamento_completo(A, b, retornar_passos=False, mostrar_matrizes=False, mostrar_permutacao=False, **kwargs):
    inicio = time.time()
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    passos = {"matrizes": [], "acoes": [], "col_permutacao": None}

    if not eh_quadrada(A) or A.shape[0] != b.shape[0]:
        status = "ERRO: Matriz A não é quadrada ou ordem inconsistente com b."
        return _empacotar_retorno(None, 0.0, status, passos, retornar_passos)

    n = len(b)
    M = np.hstack([A.copy(), b.reshape(-1, 1)])
    col_permutacao = list(range(n))
    if mostrar_matrizes:
        passos["matrizes"].append(("Inicial (A|b)", M.copy()))

    for i in range(n):
        sub = np.abs(M[i:, i:n])
        if sub.size == 0:
            status = "ERRO: Submatriz vazia durante pivoteamento completo."
            return _empacotar_retorno(None, time.time() - inicio, status, passos, retornar_passos)

        max_local = np.unravel_index(np.argmax(sub, axis=None), sub.shape)
        linha_pivo = i + max_local[0]
        coluna_pivo = i + max_local[1]
        passos["acoes"].append(f"Pivô absoluto em ({linha_pivo},{coluna_pivo}) na etapa {i}")

        if abs(M[linha_pivo, coluna_pivo]) < EPS:
            status = f"ERRO: Pivô zero (ou quase) na etapa {i+1}. Matriz singular."
            return _empacotar_retorno(None, time.time() - inicio, status, passos, retornar_passos)

        if linha_pivo != i:
            M[[i, linha_pivo], :] = M[[linha_pivo, i], :]
            passos["acoes"].append(f"Trocou linhas {i} <-> {linha_pivo}")
        if coluna_pivo != i:
            M[:, [i, coluna_pivo]] = M[:, [coluna_pivo, i]]
            col_permutacao[i], col_permutacao[coluna_pivo] = col_permutacao[coluna_pivo], col_permutacao[i]
            passos["acoes"].append(f"Trocou colunas {i} <-> {coluna_pivo}")

        for j in range(i + 1, n):
            multiplicador = M[j, i] / M[i, i]
            M[j, i:] -= multiplicador * M[i, i:]
            passos["acoes"].append(f"Eliminou linha {j} usando linha {i} (m={multiplicador:.6g})")

    # Retrosubstituição
    x_perm = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_perm[i] = (M[i, n] - np.dot(M[i, i + 1:], x_perm[i + 1:])) / M[i, i]

    x = np.zeros(n)
    for i_col in range(n):
        indice_original = col_permutacao[i_col]
        x[indice_original] = x_perm[i_col]

    tempo = time.time() - inicio
    status = "Sucesso (Gauss com pivoteamento completo)."
    if mostrar_permutacao:
        passos["col_permutacao"] = col_permutacao.copy()
    return _empacotar_retorno(x, tempo, status, passos, retornar_passos)

# ---------------------------------------------------------------
# Fatoração LU (sem pivoteamento)
# ---------------------------------------------------------------

def fatoracao_lu(A, b, retornar_passos=False, mostrar_matrizes=False, mostrar_LU=False, **kwargs):
    inicio = time.time()
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    passos = {"matrizes": [], "acoes": [], "L": None, "U": None}

    if not eh_quadrada(A) or A.shape[0] != b.shape[0]:
        status = "ERRO: A não é quadrada ou dimensões incompatíveis com b."
        return _empacotar_retorno(None, 0.0, status, passos, retornar_passos)

    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()

    for k in range(n):
        if abs(U[k, k]) < EPS:
            status = f"ERRO: Pivô zero em U[{k},{k}]."
            return _empacotar_retorno(None, time.time() - inicio, status, passos, retornar_passos)
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    tempo = time.time() - inicio
    status = "Sucesso (Fatoração LU sem pivoteamento)."
    if mostrar_LU:
        passos["L"], passos["U"] = L.copy(), U.copy()
    return _empacotar_retorno(x, tempo, status, passos, retornar_passos)

# ---------------------------------------------------------------
# Fatoração de Cholesky
# ---------------------------------------------------------------

def cholesky(A, b, retornar_passos=False, mostrar_matrizes=False, mostrar_L=False, **kwargs):
    inicio = time.time()
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    passos = {"matrizes": [], "acoes": [], "L": None}

    if not eh_quadrada(A) or A.shape[0] != b.shape[0]:
        status = "ERRO: A não é quadrada ou dimensões incompatíveis com b."
        return _empacotar_retorno(None, 0.0, status, passos, retornar_passos)

    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        status = "ERRO: Cholesky não aplicável — matriz não é definida positiva."
        return _empacotar_retorno(None, time.time() - inicio, status, passos, retornar_passos)

    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    x = np.zeros_like(b)
    LT = L.T
    for i in range(len(b) - 1, -1, -1):
        x[i] = (y[i] - np.dot(LT[i, i + 1:], x[i + 1:])) / LT[i, i]

    tempo = time.time() - inicio
    status = "Sucesso (Fatoração de Cholesky)."
    if mostrar_L:
        passos["L"] = L.copy()
    return _empacotar_retorno(x, tempo, status, passos, retornar_passos)

# ---------------------------------------------------------------
# Métodos iterativos — Gauss-Jacobi e Gauss-Seidel
# ---------------------------------------------------------------

def gauss_jacobi(A, b, x0=None, tol=1e-8, max_iter=100, retornar_passos=False, registrar_iteracoes=False, **kwargs):
    inicio = time.time()
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    passos = {"iteracoes": [], "acoes": []}
    n = b.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    if not eh_quadrada(A):
        status = "ERRO: A não é quadrada."
        return _empacotar_retorno(None, 0.0, status, passos, retornar_passos)

    D = np.diag(A)
    if np.any(np.abs(D) < EPS):
        status = "ERRO: Zero na diagonal — método Jacobi inválido."
        return _empacotar_retorno(None, 0.0, status, passos, retornar_passos)

    R = A - np.diagflat(D)

    for k in range(1, max_iter + 1):
        x_novo = (b - np.dot(R, x)) / D
        if registrar_iteracoes:
            passos["iteracoes"].append(x_novo.copy())
            passos["acoes"].append(f"Iteração {k}")
        if np.linalg.norm(x_novo - x, ord=np.inf) < tol:
            tempo = time.time() - inicio
            status = f"Convergiu em {k} iterações (Gauss-Jacobi)."
            return _empacotar_retorno(x_novo, tempo, status, passos, retornar_passos)
        x = x_novo.copy()

    tempo = time.time() - inicio
    status = "Atenção: não convergiu dentro do número máximo de iterações (Jacobi)."
    return _empacotar_retorno(x, tempo, status, passos, retornar_passos)


def gauss_seidel(A, b, x0=None, tol=1e-8, max_iter=100, retornar_passos=False, registrar_iteracoes=False, **kwargs):
    inicio = time.time()
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)
    passos = {"iteracoes": [], "acoes": []}
    n = b.shape[0]
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    if not eh_quadrada(A):
        status = "ERRO: A não é quadrada."
        return _empacotar_retorno(None, 0.0, status, passos, retornar_passos)

    for k in range(1, max_iter + 1):
        x_ant = x.copy()
        for i in range(n):
            soma1 = np.dot(A[i, :i], x[:i])
            soma2 = np.dot(A[i, i + 1:], x_ant[i + 1:])
            if abs(A[i, i]) < EPS:
                status = f"ERRO: Zero na diagonal (linha {i})."
                return _empacotar_retorno(None, time.time() - inicio, status, passos, retornar_passos)
            x[i] = (b[i] - soma1 - soma2) / A[i, i]

        if registrar_iteracoes:
            passos["iteracoes"].append(x.copy())
            passos["acoes"].append(f"Iteração {k}")
        if np.linalg.norm(x - x_ant, ord=np.inf) < tol:
            tempo = time.time() - inicio
            status = f"Convergiu em {k} iterações (Gauss-Seidel)."
            return _empacotar_retorno(x, tempo, status, passos, retornar_passos)

    tempo = time.time() - inicio
    status = "Atenção: não convergiu dentro do número máximo de iterações (Gauss-Seidel)."
    return _empacotar_retorno(x, tempo, status, passos, retornar_passos)

# ---------------------------------------------------------------
# Mapeamento usado pela interface gráfica (GUI)
# ---------------------------------------------------------------

METODOS = {
    "Gauss sem pivoteamento": eliminacao_gauss,
    "Gauss com pivoteamento parcial": pivoteamento_parcial,
    "Gauss com pivoteamento completo": pivoteamento_completo,
    "Fatoração LU": fatoracao_lu,
    "Fatoração de Cholesky": cholesky,
    "Método iterativo - Gauss-Jacobi": gauss_jacobi,
    "Método iterativo - Gauss-Seidel": gauss_seidel,
}
